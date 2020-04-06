"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""This file extends the dataset_description.json with relevant metadata and the segmentation map.
"""

import argparse
import pydicom as dicom
import pathlib
from fexp.utils.io import read_json, write_json
from fexp.readers import read_image
from fexp.utils.bbox import bounding_box
from tqdm import tqdm

# TODO: Add description check
# def __description_checksum(self):
#     # https://stackoverflow.com/a/42148379
#     checksum = 0
#     for item in self.dataset_description.items():
#         c1 = 1
#         for _ in item:
#             c1 = zlib.adler32(bytes(repr(_), 'utf-8'), c1)
#         checksum = checksum ^ c1
#     return checksum


def read_metadata(dicom_fn):
    """Extracts the following metadata:

    (0008,0070) - Manufacturer
    (0008,1090) - ManufacturerModelName
    """
    dicom_obj = dicom.read_file(str(dicom_fn), stop_before_pixels=True)
    tags = ['Manufacturer', 'ManufacturerModelName']
    output = {tag: getattr(dicom_obj, tag, '').strip() for tag in tags}
    return output


def main():
    args = parse_args()
    dataset_description = read_json(args.dataset_description)

    for patient in tqdm(dataset_description):
        for study in dataset_description[patient]:
            for image_dict in dataset_description[patient][study]:
                del image_dict['annotation']
                image_fn = pathlib.Path(image_dict['image'])
                del image_dict['image']
                image_dict['filename'] = str(image_fn)
                image_dict['label'] = str(pathlib.Path(image_fn.parent) / str(image_fn.stem + '_mask.nrrd'))
                try:
                    image_dict['bbox'] = compute_bounding_box(args.input_dir / image_dict['label'])
                except IndexError:
                    tqdm.write(f"Fail: {image_dict['label']}")

                image_dict['metadata'] = read_metadata(args.input_dir / image_fn)

    write_json(args.output_path / 'dataset_description.json', dataset_description)
    print(f'Wrote description to {args.output_path}.')


def compute_bounding_box(label_fn):
    # TODO: Better building of cache names.
    label_arr = read_image(label_fn, force_2d=True, no_metadata=True)
    bbox = bounding_box(label_arr)

    # Classes cannot be collated in the standard pytorch collate function.
    return [int(_) for _ in list(bbox)]



def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Extend dataset description with manufacturer name, device type and window and level values',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'input_dir', type=pathlib.Path,
        help="Directory to data",
    )
    parser.add_argument(
        'dataset_description', default=None, type=pathlib.Path,
        help="""JSON file describing imported data structure.

        We assume a dictionary structure patient_id -> study_id -> [{'image': ..., 'annotation': ...}]
        """)

    parser.add_argument(
        'output_path', default=None, type=pathlib.Path,
        help="""Directory to write `dataset_description.json` to which describes the exported data structure.

        We assume a dictionary structure patient_id -> study_id -> [{'image': ..., 'annotation': ...}]
        """)

    return parser.parse_args()


if __name__ == '__main__':
    main()
