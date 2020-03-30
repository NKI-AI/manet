"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from manet.utils.dicom import DICOM_MODALITY_TAG, DICOM_VOI_LUT_FUNCTION, DICOM_VOI_LUT_SEQUENCE, DICOM_WINDOW_CENTER, \
    DICOM_WINDOW_CENTER_WIDTH_EXPLANATION, DICOM_WINDOW_WIDTH, DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP, \
    DICOM_PATIENT_ORIENTATION, DICOM_LATERALITY, DICOM_IMAGE_LATERALITY, DICOM_VIEW_POSITION, \
    DICOM_PHOTOMETRIC_INTERPRETATION
from manet.utils.image import MammogramImage
from fexp.readers import read_image

import logging
logger = logging.getLogger(__name__)


def read_mammogram(filename):
    """
    Read mammograms in dicom format. Attempts to read correct DICOM LUTs.

    Parameters
    ----------
    filename : pathlib.Path or str

    Returns
    -------
    fexp.image.MammogramImage
    """
    extra_tags = [DICOM_MODALITY_TAG, DICOM_VOI_LUT_FUNCTION, DICOM_VOI_LUT_SEQUENCE,
                  DICOM_LATERALITY, DICOM_IMAGE_LATERALITY, DICOM_VIEW_POSITION,
                  DICOM_WINDOW_WIDTH, DICOM_WINDOW_CENTER, DICOM_WINDOW_CENTER_WIDTH_EXPLANATION,
                  DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP, DICOM_PATIENT_ORIENTATION, DICOM_PHOTOMETRIC_INTERPRETATION]

    image, metadata = read_image(filename, dicom_keys=extra_tags)
    dicom_tags = metadata['dicom_tags']

    modality = dicom_tags[DICOM_MODALITY_TAG]
    if not modality == 'MG':
        raise ValueError(f'{filename} is not a mammogram. Wrong Modality in DICOM header.')
    if not metadata['depth'] == 1:
        raise ValueError(f'First dimension of mammogram should be one.')

    # Remove the depth dimension
    image = image.reshape(list(image.shape)[1:])

    # Photometric Interpretation determines how to read the pixel values and if they should be inverted
    photometric_interpretation = dicom_tags[DICOM_PHOTOMETRIC_INTERPRETATION]
    if photometric_interpretation == 'MONOCHROME2':
        pass
    else:
        raise NotImplementedError(f'Photometric Interpretation {photometric_interpretation} is not implemented.')

    laterality = dicom_tags[DICOM_LATERALITY] or dicom_tags[DICOM_IMAGE_LATERALITY]
    metadata['laterality'] = laterality

    # Sometimes a horizontal flip is required:
    # https://groups.google.com/forum/#!msg/comp.protocols.dicom/X4ddGYiQOzs/g04EDChOQBwJ
    needs_horizontal_flip = dicom_tags[DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP] == 'YES'
    if laterality:
        # Check patient position
        orientation = dicom_tags[DICOM_PATIENT_ORIENTATION].split('\\')[0]
        if (laterality == 'L' and orientation == 'P') or (laterality == 'R' and orientation == 'A'):
            needs_horizontal_flip = True

    if needs_horizontal_flip:
        logger.info(f'{filename} requires a flip.')
        image = np.ascontiguousarray(np.fliplr(image))

    del metadata['depth']
    del metadata['direction']
    del metadata['origin']

    voi_lut_function = dicom_tags[DICOM_VOI_LUT_FUNCTION] if dicom_tags[DICOM_VOI_LUT_FUNCTION] else 'LINEAR'

    return MammogramImage(image, filename, metadata, voi_lut_function=voi_lut_function,
                          view=dicom_tags[DICOM_VIEW_POSITION], laterality=laterality)

