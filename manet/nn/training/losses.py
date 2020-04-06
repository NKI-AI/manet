"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch


def build_losses(use_classifier=False):
    loss_fns = [torch.nn.CrossEntropyLoss(weight=None, reduction='mean')]

    if use_classifier:
        loss_fns += torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

    return loss_fns