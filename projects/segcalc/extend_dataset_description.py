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
from tqdm import tqdm


def read_metadata(dicom_fn):
    """Extracts the following metadata:

    (0008,0070) - Manufacturer
    (0008,1090) - ManufacturerModelName
    (0028,1056) - VOILUTFunction
    """
    dicom_obj = dicom.read_file(str(dicom_fn), stop_before_pixels=True)
    tags = ['Manufacturer', 'ManufacturerModelName', 'VOILUTFunction']
    output = {tag: getattr(dicom_obj, tag, '').strip() for tag in tags}
    if hasattr(dicom_obj, 'WindowCenter') and hasattr(dicom_obj, 'WindowWidth'):
        if type(dicom_obj.WindowCenter) == dicom.multival.MultiValue:
            output['WindowCenter'] = list(dicom_obj.WindowCenter)
            output['WindowWidth'] = list(dicom_obj.WindowWidth)
        else:
            output['WindowCenter'] = float(dicom_obj.WindowCenter)
            output['WindowWidth'] = float(dicom_obj.WindowWidth)
    if hasattr(dicom_obj, 'VOILUTSequence'):
        output['VOILUTSequence'] = len(dicom_obj.VOILUTSequence)
        #sequences = {}
        #for elem in range(0, len(dicom_obj.VOILUTSequence)):
        #    sequences[elem] = dict(dicom_obj.VOILUTSequence[elem])
        #output['VOILUTSequence'] = sequences

    return output

def main():
    args = parse_args()
    dataset_description = read_json(args.dataset_description)

    for patient in tqdm(dataset_description):
        study = dataset_description[str(patient)]
        for image_dict in range(0, len(study)):
            image_fn = pathlib.Path(study[image_dict]['filename'])
            study[image_dict]['metadata'] = read_metadata(args.input_dir / image_fn)

    write_json(args.output_path / 'dataset_description.json', dataset_description)
    print(f'Wrote description to {args.output_path}.')


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