"""
Copyright (c) Nikita Moriakov and Jonas Teuwen
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
import pathlib
import hashlib
import numpy as np
import zlib

from torch.utils.data import Dataset
from manet.utils.readers import read_image

from fexp.utils.bbox import bounding_box
from fexp.utils.io import read_json, write_json, read_list, write_list

logger = logging.getLogger(__name__)

class MammoDataset(Dataset):
    def __init__(self, dataset_description, data_root, transform=None, cache_dir='/tmp'):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)

        self.data_root = data_root
        self.transform = transform
        self.cache_dir = pathlib.Path(cache_dir)

        self.filter_negatives = True
        self.use_bounding_boxes = True

        if isinstance(dataset_description, (str, pathlib.Path)):
            self.logger.info(f'Loading dataset description from file {dataset_description}.')
            dataset_description = read_json(dataset_description)
        self.dataset_description = dataset_description

        self.data = []
        labels = []

        self._cache_valid = True
        self.validate_cache()  # Pass

        for path in self.dataset_description:
            data_cache = self.cache_dir / hashlib.sha224(str(path).encode()).hexdigest()
            if data_cache.exists() and self._cache_valid:
                curr_data_cache = read_json(data_cache)
                print(f'Pulling directory {path} from cache.')
                self.data.append(curr_data_cache)
            else:
                self.logger.debug(f'Parsing directory {path}.')
                for image_dict in self.dataset_description[path]:
                    #curr_data_dict = {'case_path': path, 'image_fn': pathlib.Path(image_dict['filename'])}
                    curr_data_dict = {'case_path': path, 'image_fn': image_dict['filename'], 'class': image_dict['DCIS_stage']}
                    if self.filter_negatives and 'label' in image_dict:
                        label_fn = pathlib.Path(image_dict['label'])
                        label_fns = image_dict['label'] #because pathlib.path not json serializable
                        labels.append(label_fns)
                        curr_data_dict['label_fn'] = label_fns

                        if self.use_bounding_boxes:
                            try:
                                bbox = self.compute_bounding_box(label_fn)
                                curr_data_dict['bbox'] = [int(i) for i in bbox]
                            except IndexError:
                                self.logger.error(f'Cannot compute bounding box of {label_fn}.')
                                continue
                        self.data.append(curr_data_dict)
                        write_json(data_cache, curr_data_dict)
                        write_list('list_labels', labels)
                    else:
                        NotImplementedError()

        self.logger.info(f'Loaded dataset of size {len(self.data)}.')

    def compute_bounding_box(self, label_fn):
        # TODO: Better building of cache names.
        bbox_cache = self.cache_dir / hashlib.sha224(str(label_fn).encode()).hexdigest()
        if bbox_cache.exists() and self._cache_valid:
            bbox = read_json(bbox_cache)[str(label_fn)]
        else:
            self.logger.debug(f'Computing bounding box for {label_fn}.')
            label_arr = read_image(self.data_root / label_fn, force_2d=True, no_metadata=True)
            bbox = bounding_box(label_arr)
            write_json(bbox_cache, {str(label_fn): bbox.bbox.tolist()})

        # Classes cannot be collated in the standard pytorch collate function.
        return list(bbox)

    def validate_cache(self):
        # This function checks if the dataset description has changed, if so, the whole cache is invalidated.
        # Maybe this is a bit too strict, but this can be improved in future versions.
        cache_checksum = self.cache_dir / 'cache_checksum'
        current_checksum = self.__description_checksum()
        if not cache_checksum.exists():
            self._cache_valid = False
            write_list(cache_checksum, [current_checksum])
        else:
            checksum = int(read_list(cache_checksum)[0])
            #self._cache_valid = True if current_checksum == checksum else False
            if current_checksum == checksum:
                self._cache_valid = True
            else:
                self._cache_valid = False

    def __description_checksum(self):
        # https://stackoverflow.com/a/42148379
        checksum = 0
        for item in self.dataset_description.items():
            c1 = 1
            for _ in item:
                c1 = zlib.adler32(bytes(repr(_), 'utf-8'), c1)
            checksum = checksum ^ c1
        return checksum

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if not self.filter_negatives or not self.use_bounding_boxes:
            raise NotImplementedError()

        image_fn = self.data_root / pathlib.Path(data_dict['image_fn'])
        label_fn = self.data_root / pathlib.Path(data_dict['label_fn'])
        bbox = data_dict['bbox']
        stage = data_dict['class']

        image = read_image(image_fn, force_2d=True, no_metadata=True, dtype=np.float32)[np.newaxis, ...]
        mask = read_image(label_fn, force_2d=True, no_metadata=True, dtype=np.int64)

        sample = {
            'image': image,
            'mask': mask,
            'bbox': bbox,
            'image_fn': str(image_fn),  # Convenient for debugging errors in file loading
            'label_fn': str(label_fn),
            'class': stage
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
