# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import logging
from manet.sys import multi_gpu

logger = logging.getLogger(__name__)


def np_expand_dims(x, dims):
    t = np.copy(x)
    for val in dims:
        t = np.expand_dims(t, axis=val)
    return t


def np_complex2real(x, reorder=False):
    t = np.stack([x.real, x.imag], axis=-1)
    if not reorder:
        return t
    else:
        last_axis = len(t.shape) - 1
        perm = tuple([last_axis] + [val for val in range(len(t.shape)) if val != last_axis])
        return np.transpose(t, perm) 


def np_channel_reorder(x, in_ch_axis=None):
    if in_ch_axis is not None and in_ch_axis > 0:
        perm = tuple([in_ch_axis] + [val for val in range(len(x.shape)) if val != in_ch_axis])
        return np.transpose(x, perm)
    elif in_ch_axis is None:
        return np.expand_dims(x, axis=0)
    return x


def np_tensor_stat(x):
    return (np.min(x), np.max(x), np.mean(x), np.std(x))


def torch_real2complex(x):
    return torch.cat([x, torch.zeros_like(x)], dim=-1)


def torch_mul_real_complex(x, y):
    real = x * y[...,0]
    imag = x[...] * y[...,1]
    return torch.cat([real, imag], dim=-1)


def tensor_complex_multiplication(x, y):
    """
    Multiplies two complex-valued tensors.,
    The last axis denote the real and imaginary parts respectively.

    Parameters
    ----------
    x : torch.Tensor
        Input data
    y : torch.Tensor
        Input data

    Returns
    -------
    torch.Tensor
    """
    assert (x.shape[-1] == 2) and (y.shape[-1] == 2),\
        'Last axis has to denote the complex and imaginary part and should therefore be 2.'

    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.cat([real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)], dim=-1)


def torch_complex_abs_squared(x):
    return x[..., 0]*x[..., 0] + x[..., 1]*x[..., 1]


def torch_complex_abs(x):
    return torch.sqrt(x[..., 0]*x[..., 0] + x[..., 1]*x[..., 1])


def tensor_complex_conjugate(data):
    """
    Compute the complex conjugate of a torch tensor where the last axis denotes the real and complex part.

    Parameters
    ----------
    data : torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    assert data.shape[-1] == 2, 'Last axis has to denote the complex and imaginary part and should therefore be 2.'

    x = data.clone()  # TODO: Verify if clone is required.
    x[..., 1] = x[..., 1] * -1.0
    return x


def torch_complex2real(x):
    raise NotImplementedError()


def reduce_tensor_dict(tensors_dict):
    """
    Reduce the tensor dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    tensors_dict, after reduction.
    """
    world_size = multi_gpu.get_world_size()
    if world_size <= 1:
        return tensors_dict
    with torch.no_grad():
        tensor_names = []
        all_tensors = []
        for k in sorted(tensors_dict.keys()):
            tensor_names.append(k)
            all_tensors.append(tensors_dict[k])
        all_tensors = torch.stack(all_tensors, dim=0)
        torch.distributed.reduce(all_tensors, dst=0)
        if torch.distributed.get_rank() == 0:
            # Only accumulate in main process
            all_tensors /= world_size
        reduced_tensor_dict = {k: v for k, v in zip(tensor_names, all_tensors)}
    return reduced_tensor_dict


def tensor_to_complex_numpy(data):
    """
    Converts a complex pytorch tensor to a complex numpy array.
    The last axis denote the real and imaginary parts respectively.

    Parameters
    ----------
    data : torch.Tensor
        Input data


    Returns
    -------
    Complex valued np.ndarray
    """
    assert data.shape[-1] == 2, 'Last axis has to denote the complex and imaginary part and should therefore be 2.'
    # TODO: Check device and detaching from computation graph
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]
