"""
Copyright (c) Nikita Moriakov and Jonas Teuwen
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import hashlib
import itertools
import json
import logging
import os
import re
import h5py
import SimpleITK as sitk
import collections
import numpy as np
from random import shuffle
from config.base_config import cfg
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from manet.utils.bbox import extend_bbox, crop_to_bbox
from manet.nn.unet.resblockunet import resblockunet_shape_in
from manet.sys.io import fn_parser
from manet.data.augmentations import RandomElasticTransform, RandomGammaTransform, RandomZoomTransform, \
    RandomFlipTransform, RandomShiftTransform, RandomGaussianNoise
from manet.sys.io import read_list, read_json

logger = logging.getLogger(__name__)


class MammoDataset(Dataset):
    def __init__(self, type_dataset, source, cache_dir='/input/cache', in_shp=(21, 512, 512), debug=False):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        assert type_dataset in {'train', 'val', 'test'}, 'Choose one of `train`, `val`, `test`.'
        self.type_dataset = type_dataset
        os.makedirs(cache_dir, exist_ok=True)
        self.logger.info(f'Loading list {os.path.join(source, type_dataset)}.lst')
        self.pids = set(read_list(os.path.join(source, type_dataset + '.lst')))
        if debug:
            self.pids = set(list(self.pids)[:1])

        self.source = source
        self.logger.info(f'Loading dataset description {os.path.join(source, "dataset.json")}')
        self.descr = read_json(os.path.join(source, 'dataset.json'))
        self.entries = [entry for entry in self.descr if entry['pid'] in self.pids]
        self.patient_index = {}

        # Make GT segmentations
        pdict = {}
        exp = "highresLesion_(\w+)_(\w+)_lesionSegmentation"
        cexp = re.compile(exp)
        for eidx, ent in enumerate(self.entries):
            t = fn_parser(ent['segmentation'], cexp, ['lesion', 'auth'])
            key = ent['pid'] + '_' + ent['sid']
            if key not in pdict:
                pdict[key] = {}
            if t['lesion'] not in pdict[key]:
                pdict[key][t['lesion']] = []
            if 'Albert' not in ent['segmentation']:
                pdict[key][t['lesion']].append((eidx, ent['segmentation'], t['auth']))

        for key in pdict:
            is_radiologist = {lid: len([val for val in pdict[key][lid] if val[2] == 'Radiologist']) > 0 for lid in
                              pdict[key]}
            for lid in pdict[key]:
                if is_radiologist[lid]:
                    # Use (corrected) radiologist annotation if available
                    pdict[key][lid] = [val for val in pdict[key][lid] if val[2] == 'Radiologist']
                    if len([val for val in pdict[key][lid] if 'corrected_corrected' in val[1]]) > 0:
                        pdict[key][lid] = [val for val in pdict[key][lid] if 'corrected_corrected' in val[1]]
                    elif len([val for val in pdict[key][lid] if 'corrected' in val[1]]) > 0:
                        pdict[key][lid] = [val for val in pdict[key][lid] if 'corrected' in val[1]]
                else:
                    pass  # Use Suzan's annotation

            # Merge consensus annotations for different lesions
            all_radiologist = all([is_radiologist[key] for key in is_radiologist])
            logger.info(f'Patient/study {key}, all radiologist annotation available {all_radiologist}.')
            self.patient_index[key] = []
            # If radiologist annotation is available, merge it and use as GT
            if all_radiologist:
                for lid in pdict[key]:
                    self.patient_index[key] += [val[0] for val in pdict[key][lid] if val[2] == 'Radiologist']
            else:
                # Merge all annotations from Suzan/Radiologist
                for lid in pdict[key]:
                    self.patient_index[key] += [val[0] for val in pdict[key][lid]]


        self.in_shp = in_shp
        self.cache_dir = cache_dir
        self.patches = []
        self.base_weights = None
        self.h5_all()

        #key = np.random.random()
        #shuffle(self.entries, random=(lambda: key))
        #shuffle(self.entry_weights, random=(lambda: key))
        self.logger.info('Dataset {} patch count: {}'.format(type_dataset, len(self.patches)))

        all_augs = [None, RandomGammaTransform, RandomShiftTransform, RandomZoomTransform, RandomElasticTransform,
                    RandomGaussianNoise, RandomFlipTransform]
        self.augs = [val for val in all_augs if val is None or val.__name__ in cfg.UNET.AUG_LST]

    def cache_all(self):
        num_pos_patch = 0
        for eidx, ent in enumerate(self.entries):
            self.logger.info(f'Caching {ent["segmentation"]}')
            patientid = ent['pid']
            sitk_img_s = sitk.GetArrayFromImage(sitk.ReadImage(ent['segmentation']))
            sitk_img = sitk.GetArrayFromImage(sitk.ReadImage(ent['volume']))
            vmean, vperc = np.mean(sitk_img), np.percentile(sitk_img, 95) + 0.00001
            img_shape = sitk_img.shape
            for i in range(img_shape[0]):
                crop_shape = (1, img_shape[1], img_shape[2])
                patch_coord = (i, 0, 0)
                bbox_patch_out = [*patch_coord, *crop_shape]
                bbox_patch_out_ex, src_bbox = extend_bbox(bbox_patch_out, np.array([crop_shape[0], self.in_shp[1],
                                                                                    self.in_shp[2]]) - np.array(crop_shape),
                                                          retrieve_original=True)
                bbox_patch_in = extend_bbox(bbox_patch_out, np.array(self.in_shp) - np.array(crop_shape))
                patch_input = crop_to_bbox(sitk_img, bbox_patch_in).astype(np.uint16)
                patch_seg = crop_to_bbox(sitk_img_s, bbox_patch_out_ex).astype(np.uint8)
                num_pos_vox = np.sum(patch_seg)
                if num_pos_vox > 0:
                    num_pos_patch += 1
                np.save(os.path.join(self.cache_dir, f'vol_{patientid}_{eidx}_{i}'), patch_input)
                np.save(os.path.join(self.cache_dir, f'seg_{patientid}_{eidx}_{i}'), patch_seg)

                patch = {'eidx': eidx, 'pidx': i, 'pid': ent['pid'], 'sid': ent['sid'], 'vmean': vmean, 'vperc': vperc,
                         'src_volume': ent['volume'],
                         'src_segmentation': ent['segmentation'],
                         'src_bbox': np.array(src_bbox),
                         'pos_vox': num_pos_vox,
                         'volume': os.path.join(self.cache_dir, f'vol_{patientid}_{eidx}_{i}'),
                         'segmentation': os.path.join(self.cache_dir, f'seg_{patientid}_{eidx}_{i}')}
                self.patches.append(patch)
        num_neg_patch = len(self.patches) - num_pos_patch
        weight_lst = [cfg.POS_SAMPLE_WEIGHT / num_pos_patch if patch['pos_vox'] > 0 else
                      (1.0 - cfg.POS_SAMPLE_WEIGHT) / num_neg_patch for patch in self.patches]
        self.base_weights = np.array(weight_lst)

    def h5_all(self):
        num_pos_patch = 0
        for key in self.patient_index:
            if len(self.patient_index[key]) == 0:
                continue
            idx_0 = self.patient_index[key][0]
            ent_0 = self.entries[idx_0]
            patientid, studyid = ent_0['pid'], ent_0['sid']

            # Pad and convert volume into h5
            sitk_img = sitk.ReadImage(ent_0['volume'])
            np_img = sitk.GetArrayFromImage(sitk_img)
            vmean, vmin = np.mean(np_img), np.min(np_img)
            np_img = np_img - vmin
            vperc = (np.percentile(np_img, 95.0) + 0.00001)
            logger.info(f'Patient {patientid}, study {studyid}: min {vmin} mean {vmean}, 95% {vperc}')
            with h5py.File(os.path.join(self.cache_dir, f'vol_{patientid}_{studyid}.h5'), 'w') as hf:
                hf.create_dataset(f'volume', data=np.pad(np_img, [(10, 10), (0, 0), (0, 0)], 'constant'))

            # Prepare segmentations
            seg_buffer = np.zeros_like(np_img)
            for eidx in self.patient_index[key]:
                ent = self.entries[eidx]
                sitk_img_s = sitk.GetArrayFromImage(sitk.ReadImage(ent['segmentation']))
                # fix one flipped segmentation
                if ('02000948' in patientid and 'Suzan' in ent['segmentation']) or (
                        '02000798' in patientid and '05' in studyid):
                    sitk_img_s = sitk_img_s[::-1, ...]
                seg_buffer = seg_buffer + sitk_img_s
            seg_buffer = (seg_buffer > 0).astype(np.uint8)
            with h5py.File(os.path.join(self.cache_dir, f'seg_{patientid}_{studyid}.h5'), 'w') as hf:
                hf.create_dataset(f'segmentation', data=seg_buffer)

            img_shape = np_img.shape
            for i in range(img_shape[0]):
                num_pos_vox = np.sum(seg_buffer[i, ...])
                if num_pos_vox > 0:
                    num_pos_patch += 1
                patch = {'eidx': idx_0, 'pidx': i, 'vmean': vmean, 'vperc': vperc, 'vmin': vmin,
                         'src_origin': np.array(sitk_img.GetOrigin()),
                         'src_direction': np.array(sitk_img.GetDirection()),
                         'src_spacing': np.array(sitk_img.GetSpacing()),
                         'src_volume': ent_0['volume'],
                         'pos_vox': num_pos_vox,
                         'volume': os.path.join(self.cache_dir, f'vol_{patientid}_{studyid}.h5'),
                         'segmentation': os.path.join(self.cache_dir, f'seg_{patientid}_{studyid}.h5')}
                self.patches.append(patch)

        num_neg_patch = len(self.patches) - num_pos_patch
        self.logger.info(f'Patches {len(self.patches)}: {num_pos_patch} positive, {num_neg_patch} negative.')
        weight_lst = [0.5 / num_pos_patch if patch['pos_vox'] > 0 else 0.5 / num_neg_patch
                      for patch in self.patches]
        self.base_weights = np.array(weight_lst)

    def __len__(self):
        return len(self.patches)

    def get_weights(self, losses, percentile=0.1, scale=2.0):
        idxloss = list(enumerate(losses))
        idxloss = sorted(idxloss, key=(lambda v: -v[1]))
        new_weights = np.array(self.base_weights)
        for idx in range(int(len(idxloss) * percentile)):
            nidx, _ = idxloss[idx]
            new_weights[nidx] *= scale
            new_weights = new_weights / np.sum(new_weights)
        return new_weights
                                                                                    
    def __getitem__(self, idx):
        centry = self.patches[idx]
        volfn = centry['volume']
        segfn = centry['segmentation']
        normalizer = centry['vperc']
        i = centry['pidx']

        with h5py.File(volfn, 'r') as hf:
            img = hf['volume'][i:i + 21, ...].astype(np.float32) / normalizer
        dx, dy = self.in_shp[1] - img.shape[1], self.in_shp[2] - img.shape[2]
        img = np.pad(img, [(0, 0), (dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)], 'constant')

        if segfn:
            with h5py.File(segfn, 'r') as hf:
                mask = hf['segmentation'][i, ...].astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            mask = np.pad(mask, [(0, 0), (dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2)], 'constant')
        else:
            mask = None
        sample = {}
        ctransform = []
        if self.type_dataset.upper() == 'TRAIN':
            ctransform = [val(in_shape=img.shape) for val in np.random.choice(self.augs, cfg.UNET.NUM_AUG, replace=False,
                                                            p=np.ones(len(self.augs), dtype=np.float32) / len(self.augs))
                          if val is not None]
        for trans in ctransform:
            if trans is not None:
                img = trans.apply(img)
                if (mask is not None) and trans.mask:
                    mask = trans.apply(mask)

        if mask is not None:
            sample['mask'] = mask
        sample['image'] = np.expand_dims(img, axis=0)
        sample['src_bbox'] = np.array([0, 0, dx // 2, dx - dx // 2, dy // 2, dy - dy // 2]).astype(np.float32)
        for key in ['src_origin', 'src_direction', 'src_spacing']:
            sample[key] = centry[key].astype(np.float32)
        sample['src_volume'] = os.path.basename(centry['src_volume'])
        sample['slice_idx'] = i
        sample['patch_idx'] = idx
        return sample


def uset_collate(batch):
    """Puts each data field into a object with outer dimension batch size
    In a dictionary adds `_batch` to the key.

    """
    # The keys which can be collated
    mr_keys = ['image', 'mask', 'class']
    # All images should have the same keys
    mr_keys = [k for k in mr_keys if k in batch[0]]
    # The USet outputs a dictionary.
    # It is convenient to not convert the bbox list to a tensor.
    if isinstance(batch[0], collections.Mapping):
        out = {}
        for key in mr_keys:
            # if statement: might be the case that 'weight' is not in the
            # dictionary.
            temp_collate = [d[key] for d in batch]
            if key in ['bbox', 'patch_bbox', 'has_lesion', 'center_of_mass']:
                out[key + '_batch'] = temp_collate
            else:
                out[key + '_batch'] = default_collate(temp_collate)
        return out
    else:
        return default_collate(batch)




