# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import subprocess
import logging
import SimpleITK as sitk
import tifffile as tiff

logger = logging.getLogger(__name__)


def link_data(source, target):
    paths = [source, target]
    logger.info('Rsyncing data...')
    data_cmd = ["/usr/bin/rsync", "-am", "--stats"]
    logger.info('Executing {}'.format(data_cmd + paths))
    subprocess.call(data_cmd + paths)


def save_volume(data, fn, dest, scaling=(0.0, 1.0), fmt='tif'):
    assert len(data.shape) in [3, 4], f'Data shape {data.shape} for volume {fn} not understood.'
    assert fmt in ['tif', 'nrrd'], f'Volume format {fmt} not supported.'

    if not fn.endswith(fmt):
        fn = fn + '.' + fmt
    if len(data.shape) == 4:
        # Take zero channel only
        data = data[0, ...]

    if fmt == 'tif':
        tiff.imsave(os.path.join(dest, fn), scaling[0] + (data / scaling[1]))
    elif fmt == 'nrrd':
        img = sitk.GetImageFromArray(scaling[0] + (data / scaling[1]))
        sitk.WriteImage(img, os.path.join(dest, fn))


def fn_parser(fn, expr, param):
    m = expr.search(fn)
    if m is None:
        return None
    groups = m.groups()
    t = {}
    for i, key in enumerate(param):
        t[key] = groups[i]
    return t
