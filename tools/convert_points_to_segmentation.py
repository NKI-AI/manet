"""
 Copyright (c) Nikita Moriakov and Jonas Teuwen

 This source code is licensed under the MIT license found in the
 LICENSE file in the root directory of this source tree.
 """

"""
Convert point annotations of calcifications to segmentations by snapping to the highest intensity point in the 
neighborhood. Subsequently these regions are segmentated by applying an Otsu threshold, removing of small holes and 
objects and computing the complex hull.
"""

import skimage
import skimage.draw
import skimage.feature
import skimage.segmentation
import skimage.measure
import numpy as np
import pathlib
import argparse

from fexp.plotting import plot_2d
from fexp.readers import read_mammogram
from fexp.utils.io import read_json, read_list
from fexp.utils.bbox import crop_to_bbox
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.morphology import convex_hull_image
from tqdm import tqdm


def largest_cc(segmentation):
    """Compute largest component of segmentation.

    Taken from https://stackoverflow.com/a/54449580
    """
    labels = skimage.measure.label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg = list(zip(unique, counts))[1:]  # the 0 label is by default background so take the rest
    try:
        largest = max(list_seg, key=lambda x: x[1])[0]
    except:
        pass
    labels_max = (labels == largest).astype(int)
    return labels_max


def find_new_center(image, points, box_radius=6):
    """
    Given point annotations find the new center by looking for the point with maximal intensity in a radius.

    Parameters
    ----------
    image : np.ndarray
    points : np.ndarray
    box_radius : int

    Returns
    -------
    np.ndarray
    """
    new_points = []
    for idx, point in enumerate(points):
        # create bounding box around the annotation
        bbox = np.array(list(point - box_radius) + [2 * box_radius + 1, 2 * box_radius + 1]).astype(np.int)
        cropped_image = crop_to_bbox(image, bbox).copy()  # Weird stuff happens if you set a part of the image later
        disk = skimage.draw.circle(box_radius, box_radius, radius=box_radius)
        disk_mask = np.zeros_like(cropped_image).astype(bool)
        disk_mask[disk[0], disk[1]] = 1
        cropped_image[~disk_mask] = 0

        new_point = point - box_radius + np.unravel_index(cropped_image.argmax(), cropped_image.shape)
    new_points.append(new_point)


return np.asarray(new_points)


class Annotation:
    def __init__(self, annotations_fn, spacing):
        self.annotations_fn = pathlib.Path(annotations_fn)
        self.annotations = read_json(annotations_fn)
        self.spacing = np.asarray(spacing)

        self._annotation_spacing = np.asarray(self.annotations['annotation_spacing'])

    self.num_annotations = self.annotations.get('num_annotations', [0, 0])

    self.points_annotations = []


self.contour_annotations = []
self._parse_annotations()


def _parse_annotations(self):
    for idx in [0, 1]:
        for annotation_idx in range(self.num_annotations[idx]):
            name = f'contour_{annotation_idx}' if idx == 0 else f'points_{annotation_idx}'
            annotation_dict = self.annotations['annotations'][name]
            scaling = self._annotation_spacing / self.spacing
            points = np.fliplr(np.asarray(annotation_dict['points']) * scaling)
            bbox = np.asarray(annotation_dict['bbox'])
            bbox = np.concatenate([bbox[:2][::-1] * scaling, bbox[2:][::-1] * scaling]).astype(np.int)

            out_dict = {'points': points, 'bbox': bbox}

            if idx == 0:
                self.contour_annotations.append(out_dict)
        else:
            self.points_annotations.append(out_dict)


def crop_around_point(image, point, size):
    bbox = list((point - size).astype(np.int)) + [2 * size] * 2
    return crop_to_bbox(image, bbox), bbox


def compute_mask(image, all_coordinates, size):
    discarded = 0
    added = 0
    mask = np.zeros_like(image).astype(np.int)
    for coordinates in all_coordinates:
        cropped_image, local_bbox = crop_around_point(image, coordinates, size=size)
        if cropped_image.sum() == 0:
            print(f'Skipping {local_bbox}. Image empty.')
            continue
        local_threshold = threshold_otsu(cropped_image)
        local_mask = cropped_image > local_threshold
        local_mask = remove_small_holes(local_mask, 4)
        local_mask = remove_small_objects(local_mask, 4)

        if local_mask.sum() == local_mask.size:
            # replace by a circle
            rr, cc = skimage.draw.circle(size - 1, size - 1, radius=size)
            local_mask = np.zeros_like(local_mask)
            local_mask[rr, cc] = 1

        if local_mask.sum() == 0:
            discarded += 1
            continue

        local_mask = convex_hull_image(largest_cc(local_mask))

        if local_mask.sum() <= 4:
            discarded += 1
            continue
    if local_mask.sum() / local_mask.size >= 0.9:
        discarded += 1
        continue
    added += 1

    mask[local_bbox[0]:local_bbox[0] + local_bbox[2], local_bbox[1]:local_bbox[1] + local_bbox[3]] = local_mask


print(f'{added} added. Discarded {discarded}.')
return mask


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Create masks from point annotations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'dataset_description', default=None, type=pathlib.Path,
        help="""Text file describing data structure.

         We assume a list of the form where each line is similar to:
         full_path_to_image.dcm full_path_to_annotation.json resulting_object_name.nrrd
         """)


parser.add_argument('--num-workers', default=8, help='Number of workers to convert data.')

return parser.parse_args()


def main():
    args = parse_args()
    dataset_description = read_list(args.dataset_description)
    data_fns = [_.split(' ') for _ in dataset_description]

    # annotation_fns = pathlib.Path('/Users/jonas/PycharmProjects/small_annotations/').glob('*.json')

    annotations = []


for curr_image_fn, curr_annotations_fn, output_name in tqdm(data_fns):
    # name = annotation_fn.stem
    # curr_image_fn = f'/Users/jonas/PycharmProjects/small_features/dp{name}/original_image.dcm'
    curr_image, metadata = read_mammogram(curr_image_fn, dtype=np.float)
    # curr_annotations_fn = f'/Users/jonas/PycharmProjects/small_annotations/{name}.json'
    a = Annotation(curr_annotations_fn, metadata['spacing'])
    num_annotations = a.num_annotations[1]
    if num_annotations > 0:
        annotations.append((curr_image_fn, a))

# size to look at:
size_to_find = 0.20
for image_fn, annotation in annotations:
    print(f'Working on {image_fn}')
    image, metadata = read_mammogram(image_fn, dtype=np.float)
    coordinates = []
    for points_data in annotation.points_annotations:
        coordinates.append(points_data['points'])
    coordinates = np.vstack(coordinates)

    new_coordinates = find_new_center(image, coordinates)
size = int(np.ceil(size_to_find / metadata['spacing'][0]))
if size % 2 == 0:
    size += 1

mask = compute_mask(image, new_coordinates, size=size)
# sitk_image = sitk.GetImageFromArray(mask)
# sitk_image.SetSpacing(list(metadata['spacing']) + [1])
#
# sitk.WriteImage(sitk_image, f'curr_mask.nrrd', True)

pil_image = plot_2d(image, mask, points=new_coordinates)
pil_image.save(annotation.annotations_fn.stem + '_annotations.png')

if __name__ == '__main__':
    main()