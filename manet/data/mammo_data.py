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

from manet.utils.readers import read_mammogram
from torch.utils.data import Dataset
from fexp.readers import read_image

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

        if isinstance(dataset_description, (str, pathlib.Path)):
            self.logger.info(f'Loading dataset description from file {dataset_description}.')
            dataset_description = read_json(dataset_description)
        self.dataset_description = dataset_description

        self.data = []

        # self._cache_valid = True
        # self.validate_cache()  # Pass

        # TODO: Have bounding boxes computed elsewhere.
        for idx, patient in enumerate(self.dataset_description):
            self.logger.debug(f'Parsing patient {patient} ({idx + 1}/{len(self.dataset_description)}).')
            for study_id in self.dataset_description[patient]:
                for image_dict in self.dataset_description[patient][study_id]:
                    curr_data_dict = {'case_path': patient, 'image_fn': pathlib.Path(image_dict['filename'])}
                    if self.filter_negatives and 'label' in image_dict:
                        label_fn = pathlib.Path(image_dict['label'])
                        curr_data_dict['label_fn'] = label_fn

                        if 'bbox' not in image_dict:
                            self.logger.info(f'Patient {patient} with study {study_id} has no bounding box, skipping.')
                            continue
                        curr_data_dict['bbox'] = image_dict['bbox']

                        self.data.append(curr_data_dict)
                    else:
                        NotImplementedError()

        self.logger.info(f'Loaded dataset of size {len(self.data)}.')

    # def validate_cache(self):
    #     # This function checks if the dataset description has changed, if so, the whole cache is invalidated.
    #     # Maybe this is a bit too strict, but this can be improved in future versions.
    #     cache_checksum = self.cache_dir / 'cache_checksum'
    #     current_checksum = self.__description_checksum()
    #     if not cache_checksum.exists():
    #         self._cache_valid = False
    #         write_list(cache_checksum, [current_checksum])
    #     else:
    #         checksum = read_list(cache_checksum)[0]
    #         self._cache_valid = True if current_checksum == checksum else False
    #
    # def __description_checksum(self):
    #     # https://stackoverflow.com/a/42148379
    #     checksum = 0
    #     for item in self.dataset_description.items():
    #         c1 = 1
    #         for _ in item:
    #             c1 = zlib.adler32(bytes(repr(_), 'utf-8'), c1)
    #         checksum = checksum ^ c1
    #     return checksum

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if not self.filter_negatives:
            raise NotImplementedError()

        image_fn = self.data_root / data_dict['image_fn']
        label_fn = self.data_root / data_dict['label_fn']

        mammogram = read_mammogram(image_fn)
        mask = read_image(label_fn, force_2d=True, no_metadata=True, dtype=np.uint8)

        sample = {
            'mammogram': mammogram,
            'mask': mask,
            'bbox': data_dict['bbox'],
            'image_fn': str(image_fn),  # Convenient for debugging errors in file loading
            'label_fn': str(label_fn),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)
