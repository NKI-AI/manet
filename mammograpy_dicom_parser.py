import os
from pathlib import Path
import pydicom as dicom
from tqdm import tqdm_notebook
from pydicom.errors import InvalidDicomError
import numpy as np
import sys

import logging
logger = logging.getLogger('importer')


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
    for dicom_file in tqdm_notebook(dicoms):
        try:
            x = dicom.read_file(dicom_file, stop_before_pixels=True)
            if x.Modality == 'MG':
                laterality, view = find_laterality(x)
                if view == 'SPECIMEN':
                    logger.warning(f'{dicom_file} has view "SPECIMEN". Skipping.')
                    continue
                patient_ids.append(x.PatientID)

                mammograms[dicom_file] = {
                    'PatientID': x.PatientID,
                    'StudyInstanceUID': x.StudyInstanceUID,
                    'SeriesInstanceUID': x.SeriesInstanceUID,
                    'InstitutionName': x.InstitutionName,
                    'ViewPosition': view if view else 'NA',
                    'Laterality': laterality if laterality else 'NA',
                    'ImageType': x.ImageType
                }

            elif x.Modality in ['PR', 'SR', 'US', 'CR']:
                logger.info(f'{dicom_file} is not a mammogram, has modality {x.Modality}')
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
    return mammograms, patient_ids, failed_to_parse


def make_patient_mapping(patient_ids, encoding='10'):
    patient_ids = set(patient_ids)  # Remove duplicates
    with open('NKI_mapping.dat', 'r') as f:
        content = f.readlines()
    mapping = {k: v for k, v in [_.strip().split(' ') for _ in content if _.strip() != '']}

    new_patients = []
    start_at = len(mapping) + 1

    for patient_id in patient_ids:
        # Check if patient id already exists in dataset, and if it does, continue.
        if patient_id not in mapping:
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

    return mapping

