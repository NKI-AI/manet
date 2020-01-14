# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def synchronize():
    """
    Synchronize processes between GPUs. Wait until all devices are available.
    Function returns nothing in a non-distributed setting too.
    """
    if not torch.distributed.is_available():
        logger.info('torch.distributed: not available.')
        return

    if not torch.distributed.is_initialized():
        logger.info('torch.distributed: not initialized.')
        return

    if torch.distributed.get_world_size() == 1:
        logger.info('torch distributed: world size is 1')
        return

    torch.distributed.barrier()


def get_rank():
    """
    Get rank of the process, even when torch.distributed is not initialized.

    Returns
    -------
    int

    """
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    """
    Get number of compute device in the world, returns 1 in case multi device is not initialized.

    Returns
    -------
    int
    """
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
