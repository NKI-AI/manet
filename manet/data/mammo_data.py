"""
Copyright (c) Nikita Moriakov and Jonas Teuwen
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import pathlib
import numpy as np

from manet.utils.readers import read_mammogram
from torch.utils.data import Dataset
from fexp.readers import read_image

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

        self.class_mapping = {1: 0,
                              2: 0,
                              3: 1}

        self.data = []

        for idx, patient in enumerate(self.dataset_description):
            self.logger.debug(f'Parsing patient {patient} ({idx + 1}/{len(self.dataset_description)}).')
            curr_data_dict = {'case_path': patient}
            for study_id in self.dataset_description[patient]:
                for image_dict in self.dataset_description[patient][study_id]:


                    curr_data_dict['image_fn'] = pathlib.Path(image_dict['filename'])

                    if self.filter_negatives and 'label' in image_dict:
                        label_fn = pathlib.Path(image_dict['label'])
                        curr_data_dict['label_fn'] = label_fn

                        if 'bbox' not in image_dict:
                            self.logger.info(f'Patient {patient} with study {study_id} has no bounding box, skipping.')
                            continue
                        if 'DCIS_grade' not in image_dict:
                            self.logger.warning(f'Patient {patient} with study {study_id} has no bounding box but has a label.')
                            continue

                        if self.class_mapping:
                            class_label = self.class_mapping[image_dict['DCIS_grade']]
                        else:
                            class_label = image_dict['DCIS_grade']

                        curr_data_dict['DCIS_grade'] = class_label
                        curr_data_dict['bbox'] = image_dict['bbox']

                        self.data.append(curr_data_dict)
                    else:
                        NotImplementedError()

        self.logger.info(f'Loaded dataset of size {len(self.data)}.')

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if not self.filter_negatives:
            raise NotImplementedError()

        image_fn = self.data_root / data_dict['image_fn']
        label_fn = self.data_root / data_dict['label_fn']

        mammogram = read_mammogram(image_fn)
        mask = read_image(label_fn, force_2d=True, no_metadata=True, dtype=np.int64)  # int64 gets cast to LongTensor

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


def build_datasets(data_source):
    # Assume the description file, a training set and a validation set are linked in the main directory.
    train_list = read_list(data_source / 'training_set.txt')
    validation_list = read_list(data_source / 'validation_set.txt')
    mammography_description = read_json(data_source / 'dataset_description.json')

    training_description = {k: v for k, v in mammography_description.items() if k in train_list}
    validation_description = {k: v for k, v in mammography_description.items() if k in validation_list}

    return training_description, validation_description
