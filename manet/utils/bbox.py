# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np


# TODO: This class can be extended so boxes can be added and subtracted.
class BoundingBox(object):
    def __init__(self, bbox):
        self.bbox = np.asarray(bbox)
        self.coordinates, self.size = split_bbox(bbox)

    @property
    def center(self):
        return self.coordinates - self.size // 2

    def bounding_box_around_center(self, output_size):
        output_size = np.asarray(output_size)
        return BoundingBox(combine_bbox(self.center - output_size // 2, output_size))

    def __add__(self, x):
        self.coordinates_2, self.size_2 = x.coordinates, x.size
        new_coordinates = np.stack([self.coordinates, self.coordinates_2]).min(axis=0)
        new_size = np.stack([self.size, self.size_2]).max(axis=0)
        return BoundingBox(combine_bbox(new_coordinates, new_size))

    def __len__(self, x):
        return len(self.bbox) // 2

    def __getitem__(self, idx):
        return self.bbox[idx]


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

def add_dim(bbox, dim_sz, pre=True, coord=0):
    """Add extra dimension to bbox of size dim_sz
    """
    bbox_coords, bbox_size = split_bbox(bbox)
    if pre:
        bbox_coords = [coord] + bbox_coords.tolist()
        bbox_size = [dim_sz] + bbox_size.tolist()
    else:
        bbox_coords = bbox_coords.tolist() + [coord]
        bbox_size =  bbox_size.tolist() + [dim_sz]
    return combine_bbox(bbox_coords, bbox_size)


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


def bounding_box(mask):
    """
    Computes the bounding box of a mask
    Parameters
    ----------
    mask : array-like
        Input mask

    Returns
    -------
    Bounding box
    """
    bbox_coords = []
    bbox_sizes = []
    for idx in range(mask.ndim):
        axis = tuple([i for i in range(mask.ndim) if i != idx])
        nonzeros = np.any(mask, axis=axis)
        min_val, max_val = np.where(nonzeros)[0][[0, -1]]
        bbox_coords.append(min_val)
        bbox_sizes.append(max_val - min_val + 1)

    return combine_bbox(bbox_coords, bbox_sizes)


def crop_to_bbox(image, bbox, pad_value=0):
    """Extract bbox from images, coordinates can be negative.

    Parameters
    ----------
    image : ndarray
       nD array
    bbox : list or tuple or BoundingBox
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value : number
       if bounding box would be out of the image, this is value the patch will be padded with.

    Returns
    -------
    ndarray
    """
    if not isinstance(bbox, BoundingBox):
        bbox = BoundingBox(bbox)
    # Coordinates, size
    bbox_coords, bbox_size = bbox.coordinates, bbox.size

    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0
    r_offset = (bbox_coords + bbox_size) - np.array(image.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j
                  in zip(bbox_coords + l_offset,
                         bbox_coords + bbox_size - r_offset)]
    out = image[tuple(region_idx)]

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    patch = pad_value*np.ones(bbox_size, dtype=image.dtype)
    patch_idx = [slice(i, j) for i, j
                 in zip(l_offset, bbox_size - r_offset)]
    patch[patch_idx] = out
    return patch
