# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from manet.nn.unet.unet_fastmri_facebook import UnetModel2d


def build_model(device, use_classifier=False):
    if use_classifier:
        model = None

    else:
        model = UnetModel2d(
            1, 2, (1024, 1024), 64, 4, 0.1).to(device)

    return model
