# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np

from manet.utils.readers import read_image
from manet.utils.bbox import bounding_box
from manet.sys.io import read_json, write_json


def parse_json(path):
    mammography_data = read_json(path)
    label_paths = []
    for study in mammography_data.values():
        if 'label' in study:
            label_paths.append(study['label'])

    return label_paths


def get_stats(label_paths):
    stats = {}
    for label_fn in label_paths:
        mask, metadata = read_image(label_fn, force_2d=True)
        volume = mask.sum() * np.prod(np.array(metadata['spacing']))
        bbox = bounding_box(volume)

        stats[label_fn] = {'volume': volume, 'height': bbox[-2], 'width': bbox[-1]}

    return stats


if __name__ == '__main__':
    label_paths = parse_json('mammograms_imported.json')
    stats = get_stats(label_paths)
    write_json(stats, 'label_stats.json')
