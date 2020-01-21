"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from typing import Union


# TODO: Factor this function to exp
def clip_and_scale(
        arr: np.ndarray,
        clip_range: Union[bool, tuple, list]=False,
        source_interval: Union[bool, tuple, list]=False,
        target_interval: Union[bool, tuple, list] = False):
    """
    Clips image to specified range, and then linearly scales to the specified range (if given).

    In particular, the range in source interval is mapped to the target interval linearly,
    after clipping has been applied.

    - If clip_range is not set, the image is not clipped.
    - If target_interval is not set, only clipping is applied.
    - If source_interval is not set, the minimum and maximum values will be picked.

    Parameters
    ----------
    arr : array_like
    clip_range : tuple
        Range to clip input array to
    source_interval : tuple
       If given, this denote the original minimal and maximal values.
    target_interval : tuple
        Interval to map input values to

    Returns
    -------
    ndarray
        Clipped and scaled array.

    """
    arr = np.asarray(arr)
    if clip_range and tuple(clip_range) != (0, 0):
        if not len(clip_range) == 2:
            raise ValueError('Clip range must be two a tuple of length 2.')
        arr = np.clip(arr, clip_range[0], clip_range[1])
    if target_interval and tuple(target_interval) != (0, 0):
        if not len(target_interval) == 2:
            raise ValueError('Scale range must be two a tuple of length 2.')
        if source_interval:
            arr_min, arr_max = source_interval
        else:
            arr_min = arr.min()
            arr_max = arr.max()
        if arr_min == arr_max:
            if not arr_max == 0:
                arr = target_interval[1] * arr / arr_max
        else:
            size = target_interval[1] - target_interval[0]
            arr -= arr_min
            arr = arr / (arr_max - arr_min)
            arr *= size
            arr += target_interval[0]
    return arr