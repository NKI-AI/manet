"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pydicom as dicom
import numpy as np
import sys
import shutil
import logging
import argparse
import hashlib


from collections import defaultdict
from pydicom.errors import InvalidDicomError
from pathlib import Path


from fexp.utils.io import write_json
from fexp.readers import read_image
from fexp.utils.bbox import bounding_box

logger = logging.getLogger('mammo_importer')
logging.getLogger().setLevel(logging.INFO)


def try_copy_link(a, b, create_links):
    try:
        if create_links:
            os.symlink(a, b)
        else:
            shutil.copy(a, b)

    except FileExistsError as e:
        # Perhaps the file already exists, let's check if it is the same.
        if md5(a) == md5(b):
            return
        else:
            logger.error(
                f'Something is seriously {b} already exists, but has a different checksum from {a}.')


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def write_list(x, path):
    with open(path, 'w') as f:
        for line in x:
            f.write(line + '\n')


def find_dicoms(path):
    logger.info(f'Looking for all dicom files in {path}. This can take a while...')
    path = os.path.normpath(path)
    dicoms = []
    for root, dirs, files in os.walk(path, topdown=True):
        for d in [os.path.join(root, d) for d in dirs]:
            dcms = list(Path(d).glob('*.dcm'))
            if dcms:
                dicom_files = [str(dcm) for dcm in dcms]
                dicoms += dicom_files
            elif files:
                logger.warning(f'{d} does not contain dicom files, but {len(files)} other files.')
    logger.info(f'Found {len(dicoms)} dicom files.')
    return dicoms


def find_laterality(dcm_obj):
    for laterality_key in [(0x20, 0x62), (0x20, 0x60)]:
        laterality = dcm_obj.get(laterality_key, None)
        if laterality:
            laterality = dcm_obj[laterality_key].value.strip()
            if len(laterality.split()) == 0:  # Try harder if no laterality is found
                laterality = None
                continue
            break

    assert laterality in ['L', 'R', None]

    for view_key in [(0x18, 0x5101)]:  # @Madelon are there more relevant keys?
        view = dcm_obj.get(view_key, None)
        if view:
            view = dcm_obj[view_key].value.strip()
            if len(view.split()) == 0:  # Try harder if no view is found
                view = None
                continue
            break

    assert view in ['MLO', 'CC', 'LL', 'LM', 'ML', 'RL',
                    'XCCL', 'TAN', 'SPECIMEN', 'RMLO', 'RCC', 'LMLO', 'LCC', None], view
    if view in ['RMLO', 'RCC', 'LMLO', 'LCC']:
        view = view[1:]

    return laterality, view


def find_mammograms(dicoms):
    failed_to_parse = []
    mammograms = {}
    patient_ids = []
    bad_manufacturer = []
    too_small = []
    for dicom_file in dicoms:
        logger.info(f'Parsing {dicom_file}.')
        try:
            x = dicom.read_file(dicom_file, stop_before_pixels=True)
            if x.Modality == 'MG':
                if x.Manufacturer in ['R2 Technology, Inc.']:
                    logger.warning(f'{dicom_file} is of {x.Manufacturer}. Skipping.')
                    bad_manufacturer.append(dicom_file)
                    continue

                # Might consider removing the following as well:
                # (0008,2111) ST [Enhanced image created by Carestream Client] #  44, 1 DerivationDescription

                if x.Rows < 1500 and x.Columns < 1500:
                    logger.warning(f'{dicom_file} is too small. Skipping.')
                    too_small.append(dicom_file)
                    continue

                try:
                    if len(x.pixel_array.shape) == 3:
                        failed_to_parse.append(dicom_file)
                        logger.warning(f'Skipping TOMO.')
                        continue
                except Exception: # compressed pixel_array
                    pass

                laterality, view = find_laterality(x)
                if view == 'SPECIMEN':
                    logger.warning(f'{dicom_file} has view "SPECIMEN". Skipping.')
                    continue
                patient_ids.append(x.PatientID)

                mammograms[dicom_file] = {
                    'PatientID': x.PatientID,
                    'StudyInstanceUID': x.StudyInstanceUID,
                    'SeriesInstanceUID': x.SeriesInstanceUID,
                    'SeriesDescription': getattr(x, 'SeriesDescription', ''),
                    'InstitutionName': getattr(x, 'InstitutionName', ''),
                    'Manufacturer': getattr(x, 'Manufacturer', ''),
                    'ViewPosition': view if view else 'NA',
                    'Laterality': laterality if laterality else 'NA',
                    'ImageType': list(x.ImageType)
                }

            elif x.Modality in ['PR', 'SR', 'US', 'CR']:
                logger.info(f'{dicom_file} is not a mammogram, has modality {x.Modality}.')
            elif x.Modality == 'OT':
                logger.info(f'{dicom_file} has modality "OT" (other). Trying to parse if this is a mammogram.')
                # Check if empty
                is_empty = np.abs(x.pixel_array).sum() == 0
                if is_empty:
                    logger.warning(f'{dicom_file} has no pixel data. Skipping.')
                else:
                    logger.error(f'{dicom_file} has an uncaught exception. Exiting.')
                    sys.exit('Exiting with error. Check log.')

            else:
                logger.error(f'{dicom_file} has an uncaught modality {x.Modality}. Exiting.')
                sys.exit('Exiting with error. Check log.')

        except (AttributeError, InvalidDicomError) as e:
            logger.error(f'Failed to import {dicom_file} with error: {e}')
            failed_to_parse.append(dicom_file)

    logger.info(f'Found {len(mammograms)} and failed to parse {len(failed_to_parse)} files.')
    write_list(too_small, 'too_small.log')
    write_list(bad_manufacturer, 'bad_manufacturer.log')

    return mammograms, patient_ids, failed_to_parse


def make_patient_mapping(dest, patient_ids, encoding='10', duplicates=None):
    patient_ids = set(patient_ids)  # Remove duplicates
    if not (dest / 'NKI_mapping.dat').exists():
        logger.info('NKI_mapping.dat does not exist! Creating.')
        os.mknod(str(dest / 'NKI_mapping.dat'))

    with open(str(dest / 'NKI_mapping.dat'), 'r') as f:
        content = f.readlines()
    mapping = {k: v for k, v in [_.strip().split(' ') for _ in content if _.strip() != '']}

    new_patients = []
    start_at = len(mapping) + 1

    for patient_id in patient_ids:
        # Check if patient id already exists in dataset, and if it does, continue.
        if patient_id not in mapping:
            for duplicate_list in duplicates:
                if patient_id in duplicate_list:

            new_patients.append(patient_id)
    n_cases = len(new_patients)
    logger.info(f'{n_cases} new cases.')
    if n_cases == 0:
        return mapping

    new_ids = [f'{encoding}' + '{:7d}'.format(idx).replace(' ', '0') for idx in range(start_at, start_at + n_cases)]
    with open('NKI_mapping.dat', 'a') as f:
        for idx, line in enumerate(new_patients):
            mapping[line] = new_ids[idx]
            f.write(f'{line} {new_ids[idx]}\n')

    # Fix duplicates:
    duplicates_mapping = defaultdict(list)
    for duplicate_list in duplicates:
        begin_key = duplicate_list.pop(0)
        for other_key in duplicate_list:
            duplicates_mapping[begin_key].append(other_key)

    with open(str(dest / 'NKI_mapping.dat'), 'r') as f:
        content = f.readlines()
    mapping = {k: v for k, v in [_.strip().split(' ') for _ in content if _.strip() != '']}

    for key in duplicates_mapping:
        NKI_id = mapping[key]
        for other in duplicates_mapping[key]:
            mapping[other] = NKI_id

    # Write solution back
    with open('NKI_mapping.dat', 'w') as f:
        for key, value in mapping.items():
            f.write(f'{key} {value}\n')

    return mapping


def rewrite_structure(mammograms_dict, mapping, new_path):
    """
    Returns a dictionary of patient ids mapping to studyinstance uids, and a studyinstanceuids map to integers. The patient
    itself is cached, and when new files are added it is checked against this metadata if these files actually exist
    """

    studies_per_patient = defaultdict(list)
    uid_mapping = {}
    for fn in mammograms_dict:
        study_instance_uid = mammograms_dict[fn]['StudyInstanceUID']
        patient_id = mammograms_dict[fn]['PatientID']
        if study_instance_uid not in studies_per_patient[patient_id]:
            studies_per_patient[patient_id].append(study_instance_uid)

        path_to_patient = Path(new_path) / mapping[patient_id]
        os.makedirs(path_to_patient, exist_ok=True)
        metadata_file = path_to_patient / f'studies.dat'
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                f.write('StudyInstanceUIDs\n')

    for patient_id in studies_per_patient:
        with open(Path(new_path) / mapping[patient_id] / 'studies.dat', 'r') as f:
            study_instance_uids = [_.strip() for _ in f.readlines()[1:]]

        new_study_instance_uids = list(study_instance_uids)
        # Add new study instance UIDs to the list if they are not yet there.
        new_study_instance_uids.extend(x for x in studies_per_patient[patient_id]
                                       if x not in new_study_instance_uids)

        for idx, study_instance_uid in enumerate(new_study_instance_uids):
            if not study_instance_uid in study_instance_uids:
                with open(Path(new_path) / mapping[patient_id] / 'studies.dat', 'a') as f:
                    f.write(study_instance_uid + '\n')
            uid_mapping[study_instance_uid] = '{:2d}'.format(idx + 1).replace(' ', '0')

    return dict(studies_per_patient), uid_mapping


def create_temporary_file_structure(mammograms, patient_mapping, uid_mapping, new_path, dcis_labels=None, create_links=True):
    output = dict()
    labels_found = []

    dcis_dict = {}
    if dcis_labels:
        dcis_dict = {x.split('\t')[0].strip(): int(x.split('\t')[1].strip()) for x in dcis_labels.read_text().split('\n') if x}

    fns_added = []

    for current_mammogram_fn in mammograms:
        current_dictionary = {}

        patient_id = mammograms[current_mammogram_fn]['PatientID']

        current_dictionary['Original_PatientID'] = patient_id

        # Get DCIS label
        if patient_id in dcis_dict:
            current_dictionary['DCIS_grade'] = dcis_dict[patient_id]

        new_patient_id = patient_mapping[patient_id]

        if new_patient_id not in output:
            output[new_patient_id] = {}

        study_instance_uid = mammograms[current_mammogram_fn]['StudyInstanceUID']
        new_study_instance_uid = uid_mapping[study_instance_uid]

        if new_study_instance_uid not in output[new_patient_id]:
            output[new_patient_id][new_study_instance_uid] = []

        new_path_to_study = new_path / new_patient_id / new_study_instance_uid
        new_path_to_study.mkdir(exist_ok=True)

        current_mammogram_fn = Path(current_mammogram_fn)
        new_path_to_current_mammogram_fn = new_path_to_study / current_mammogram_fn.name
        if new_path_to_current_mammogram_fn in fns_added:
            continue  # This is a duplicate
        fns_added.append(new_path_to_current_mammogram_fn)

        current_dictionary['image'] = str(new_path_to_current_mammogram_fn.relative_to(new_path))

        # Link or copy data to its new place
        try_copy_link(current_mammogram_fn, new_path_to_current_mammogram_fn, create_links)

        # Check for label
        path_to_possible_label = Path(str(current_mammogram_fn).replace('.dcm', '-label.nrrd'))

        if path_to_possible_label.exists():
            new_path_to_label = new_path_to_study / path_to_possible_label.name
            try_copy_link(path_to_possible_label, new_path_to_label, create_links)
            current_dictionary['label'] = str(new_path_to_label.relative_to(new_path))
            try:
                bbox = compute_bounding_box(new_path_to_label)
                current_dictionary['bbox'] = bbox
            except (IndexError, ValueError) as e:
                logger.error(f"Fail bbox compute: {new_path_to_label}: {e}")

        output[new_patient_id][new_study_instance_uid].append(current_dictionary)

    return output


def compute_bounding_box(label_fn):
    # TODO: Better building of cache names.
    label_arr = read_image(label_fn, force_2d=True, no_metadata=True)  # TODO fix force_2d
    bbox = bounding_box(label_arr)

    # Classes cannot be collated in the standard pytorch collate function.
    return [int(_) for _ in list(bbox)]


def main():
    parser = argparse.ArgumentParser(description='Process dataset into convenient format.')
    parser.add_argument('path', type=Path, help='Path to dataset')
    parser.add_argument('dest', type=Path, help='Destination directory')
    parser.add_argument('--dcis-labels', type=Path, help='filename to labels filename.')
    parser.add_argument('--duplicates-list', type=Path, help='filename to duplicates.')

    parser.add_argument('--copy-data', action='store_true', help='Copy data instead of symlinking.')
    args = parser.parse_args()

    dicoms = find_dicoms(args.path)
    mammograms, patient_ids, failed_to_parse = find_mammograms(dicoms)

    write_json(args.dest / 'mammograms.json', mammograms)

    with open('failed_to_parse.log', 'a') as f:
        for line in failed_to_parse:
            f.write(line + '\n')
            
    if args.duplicates_list:
        with open(args.duplicates_list, 'r') as f:
            content = f.readlines()
        duplicates = [_.strip().split(',') for _ in content]
    else:
        duplicates = None

    patient_mapping = make_patient_mapping(args.dest, patient_ids, duplicates=duplicates)
    write_json(args.dest / 'patient_mapping.json', patient_mapping)

    studies_per_patient, uid_mapping = rewrite_structure(mammograms, patient_mapping, new_path=args.dest)

    write_json(args.dest / 'studies_per_patient.json', studies_per_patient)
    write_json(args.dest / 'uid_mapping.json', uid_mapping)

    logging.info('Writing new directory structure. This can take a while.')
    new_mammograms = create_temporary_file_structure(
        mammograms, patient_mapping, uid_mapping,
        args.dest, dcis_labels=args.dcis_labels, create_links=not args.copy_data)

    write_json(args.dest / 'dataset_description.json', new_mammograms)
    write_list(new_mammograms.keys(), args.dest / 'imported_studies.log')


if __name__ == '__main__':
    main()
