# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from fexp.utils.bbox import BoundingBox
from fexp.utils.bbox import bounding_box, crop_to_bbox


def split_bbox(bbox):
    """Split bbox into coordinates and size

    Parameters
    ----------
    bbox : tuple or ndarray. Given dimension n, first n coordinates are the starting point, the other n the size.

    Returns
    -------
    coordinates and size, both ndarrays.
    """
    bbox = np.asarray(bbox)

    ndim = int(len(bbox) / 2)
    bbox_coords = bbox[:ndim]
    bbox_size = bbox[ndim:]
    return bbox_coords, bbox_size


def combine_bbox(bbox_coords, bbox_size):
    """Combine coordinates and size into a bounding box.

    Parameters
    ----------
    bbox_coords : tuple or ndarray
    bbox_size : tuple or ndarray

    Returns
    -------
    bounding box

    """
    bbox_coords = np.asarray(bbox_coords).astype(int)
    bbox_size = np.asarray(bbox_size).astype(int)
    bbox = tuple(bbox_coords.tolist() + bbox_size.tolist())
    return bbox


def extend_bbox(bbox, extend, retrieve_original=False):
    """Extend bounding box by `extend`. Will enlarge the bounding box by extend and shift by extend // 2.
    If retrieve_original is True will returns the pair (newbbox, oldbbox) of the new and the original bbox in the
    relative coordinates of the new one.

    Parameters
    ----------
    bbox : tuple or ndarray
    extend : tuple or ndarray
    retrieve_original: boolean
    Returns
    -------
    bounding box

    """
    if not np.any(extend):
        return bbox

    bbox_coords, bbox_size = split_bbox(bbox)
    extend = np.asarray(extend)
    newbbox = combine_bbox(bbox_coords - extend // 2, bbox_size + extend)
    if retrieve_original:
        return newbbox, combine_bbox(extend // 2, bbox_size)
    else:
        return newbbox


def convert_bbox(bbox, to='xyXY'):
    """

    Parameters
    ----------
    bbox
    to : str

    Returns
    -------

    """
    if not to in ['xyXY']:
        raise NotImplementedError()

    if to == 'xyXY':
        bbox_coords, bbox_size = split_bbox(bbox)
        bbox_coords2 = np.asarray(bbox_coords) + np.asarray(bbox_size)

        bbox = np.concatenate([bbox_coords, bbox_coords2]).tolist()
    return bbox


def crop_bbox_to_shape(bbox_orig, shape):
    """

    Parameters
    ----------
    bbox_orig : tuple or ndarray
    shape : tuple

    Returns
    -------
    bounding box and offsets needed to add to original bbox to get the cropped one.

    TODO: Efficiency...

    """
    ndim = int(len(bbox_orig) / 2)
    bbox = np.array(bbox_orig)

    bbox[bbox < 0] = 0
    for idx, _ in enumerate(bbox[ndim:]):
        if _ + bbox[idx] > shape[idx]:
            bbox[idx + ndim] = shape[idx] - bbox[idx]

    return bbox


def get_random_shift_bbox(bbox, minoverlap=0.3, exclude=[]):
    """Shift bbox randomly so that all its sides have at least minoverlap fraction of intersection with originial.
    Dimension in exclude are fixed.
    Parameters
    ----------
    bbox: tuple
    minoverlap: number in (0, 1)
    exclude: tuple
    Returns
    -------
    bbox: tuple
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    deltas = [np.floor(val*minoverlap).astype(int) for val in bbox_size]
    out_coords = []
    for i, coord, delta, sz in zip(range(len(bbox_coords)), bbox_coords, deltas, bbox_size):
        if i in exclude:
            out_coords.append(coord)
        else:
            x = np.random.randint(coord - sz + delta + 1, high=(coord + sz - delta - 1), size=1)[0]
            out_coords.append(x)

    return list(out_coords) + list(bbox_size)


# def add_dim(bbox, dim_sz, pre=True, axis=0):
#     """Add extra dimension to bbox of size dim_sz
#     """
#     bbox_coords, bbox_size = split_bbox(bbox)
#     if pre:
#         bbox_coords = [axis] + bbox_coords.tolist()
#         bbox_size = [dim_sz] + bbox_size.tolist()
#     else:
#         bbox_coords = bbox_coords.tolist() + [axis]
#         bbox_size = bbox_size.tolist() + [dim_sz]
#     return combine_bbox(bbox_coords, bbox_size)


def project_bbox(bbox, exclude=[]):
    """Project bbox by excluding dimensions in exclude
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    out_coords = []
    out_size = []
    for x, d, i in zip(bbox_coords, bbox_size, range(len(bbox_coords))):
        if i not in exclude:
            out_coords.append(x)
            out_size.append(d)
    return out_coords + out_size
    

def expand_to_multiple(bbox, div=16, exclude=[]):
    """Extend bounding box so that its sides are multiples of given number, unless axis is in exclude.
    Parameters
    ----------
    bbox: list or tuple
        bbox of the form (coordinates, size)
    div: integer
        value which we want the bounding box sides to be multiples of
    exclude: list or tuple
        list of axis which are left unchanged
    Returns
    -------
    bounding box
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    extend = [int(idx not in exclude)*(div - (val % div)) for idx, val in enumerate(bbox_size)]
    return extend_bbox(bbox, extend)