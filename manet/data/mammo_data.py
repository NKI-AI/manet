"""
Copyright (c) Nikita Moriakov and Jonas Teuwen
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import logging
import pathlib
import hashlib
import numpy as np

from torch.utils.data import Dataset
from manet.sys.io import read_json, dump_json
from manet.utils.bbox import bounding_box
from manet.utils.readers import read_image


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

        self._cache_valid = True
        self.validate_cache()  # Pass

        for path in self.dataset_description:
            self.logger.debug(f'Parsing directory {path}.')
            curr_data_dict = {'case_path': path}
            for image_dict in self.dataset_description[path]:
                curr_data_dict['image_fn'] = pathlib.Path(image_dict['filename'])
                if self.filter_negatives and 'label' in image_dict:
                    label_fn = pathlib.Path(image_dict['label'])
                    if self.use_bounding_boxes:
                        bbox = self.compute_bounding_box(label_fn)
                    curr_data_dict['label_fn'] = label_fn
                    curr_data_dict['bbox'] = bbox
                    self.data.append(curr_data_dict)
                else:
                    NotImplementedError()

        self.logger.info(f'Loaded dataset of size {len(self.data)}.')

    def compute_bounding_box(self, label_fn):
        self.logger.debug(f'Computing bounding box for {label_fn}.')
        # TODO: Better building of cache names.
        bbox_cache = self.cache_dir / hashlib.sha224(str(label_fn).encode()).hexdigest()
        if bbox_cache.exists():
            bbox = read_json(bbox_cache)[str(label_fn)]

        else:
            label_arr = read_image(self.data_root / label_fn, force_2d=True, no_metadata=True)
            bbox = bounding_box(label_arr)
            dump_json(bbox_cache, {str(label_fn): bbox})

        return bbox

    def validate_cache(self):
        # This function could be used to cache things like images, etc.
        # Right now it is a dummy function.
        self._cache_valid = True

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if not self.filter_negatives or not self.use_bounding_boxes:
            raise NotImplementedError()

        image_fn = self.data_root / data_dict['image_fn']
        label_fn = self.data_root / data_dict['label_fn']
        bbox = data_dict['bbox']

        image = read_image(image_fn, force_2d=True, no_metadata=True, dtype=np.float32)[np.newaxis, ...]
        mask = read_image(label_fn, force_2d=True, no_metadata=True, dtype=np.int64)

        sample = {'image': image, 'mask': mask, 'bbox': bbox}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
