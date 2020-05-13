# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging

from manet.nn.unet.unet_fastmri_facebook import UnetModel2d
from manet.nn.unet import unet_classifier

logger = logging.getLogger(__name__)


def build_model(use_classifier=False, classifier_grad_scale=0.25):
    num_channels = 1
    num_base_filters = 32
    output_shape = (1024, 1024)
    depth = 7

    # TODO: Create config for these variables
    if use_classifier:
        logger.info(f'Using classifier model.')
        model = unet_classifier.UnetModel2dClassifier(
            num_channels, 2, 2, output_shape, num_base_filters, depth, 0.1,
            classifier_grad_scale=classifier_grad_scale)

    else:
        model = UnetModel2d(
            num_channels, 2, output_shape, num_base_filters, depth, 0.1)

    return model
