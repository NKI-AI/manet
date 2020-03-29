"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import SimpleITK as sitk
import tempfile

from manet.utils.dicom import DICOM_MODALITY_TAG, DICOM_VOI_LUT_FUNCTION, DICOM_VOI_LUT_SEQUENCE, DICOM_WINDOW_CENTER, \
    DICOM_WINDOW_CENTER_WIDTH_EXPLANATION, DICOM_WINDOW_WIDTH, DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP, \
    DICOM_PATIENT_ORIENTATION, DICOM_LATERALITY, DICOM_IMAGE_LATERALITY, DICOM_VIEW_POSITION, \
    DICOM_PHOTOMETRIC_INTERPRETATION
from manet.utils.image import MammogramImage

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
        image = np.ascontiguousarray(np.fliplr(image))

    del metadata['depth']
    del metadata['direction']
    del metadata['origin']

    voi_lut_function = dicom_tags[DICOM_VOI_LUT_FUNCTION] if dicom_tags[DICOM_VOI_LUT_FUNCTION] else 'LINEAR'

    return MammogramImage(image, filename, metadata, voi_lut_function=voi_lut_function,
                          view=dicom_tags[DICOM_VIEW_POSITION], laterality=laterality)



def read_dcm_series(path, series_id=None, filenames=False, return_sitk=False):
    """Read dicom series from a folder. If multiple dicom series are availabe in the folder,
    no image is returned. The metadata dictionary then contains the SeriesIDs which can be selected.

    Parameters
    ----------
    path : str
        path to folder containing the series
    series_id : str
        SeriesID to load
    filenames : str
        If filenames is given then series_id is ignored, and assume that there is one series and these files are loaded.
    return_sitk : bool
        If true, the original SimpleITK image will also be returned

    Returns
    -------
    metadata dictionary and image as ndarray.

    TODO
    ----
    Catch errors such as
    WARNING: In /tmp/SimpleITK-build/ITK/Modules/IO/GDCM/src/itkGDCMSeriesFileNames.cxx, line 109
    GDCMSeriesFileNames (0x4a6e830): No Series were found
    """

    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory')

    metadata = {'filenames': []}

    if filenames:
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(filenames[0])
        file_reader.ReadImageInformation()
        series_id = file_reader.GetMetaData('0020|000e')
        with tempfile.TemporaryDirectory() as tmpdir_name:
            for f in filenames:
                os.symlink(os.path.abspath(f), os.path.join(tmpdir_name, os.path.basename(f)))
            sorted_filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(tmpdir_name, series_id)
            metadata['filenames'].append(sorted_filenames)
    else:
        reader = sitk.ImageSeriesReader()
        series_ids = list(reader.GetGDCMSeriesIDs(str(path)))

        for series_id in series_ids:
            sorted_filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, series_id)
            metadata['filenames'].append(sorted_filenames)
            # TODO: Get series description.

        if len(series_ids) > 1 and not series_id:
            image = None
            metadata['series_ids'] = series_ids

            return image, metadata

    metadata['series_ids'] = series_ids
    sitk_image = sitk.ReadImage(sorted_filenames)

    metadata['filenames'] = [sorted_filenames]
    metadata['depth'] = sitk_image.GetDepth()
    metadata['spacing'] = tuple(sitk_image.GetSpacing())
    metadata['origin'] = tuple(sitk_image.GetOrigin())
    metadata['direction'] = tuple(sitk_image.GetDirection())
    data = sitk.GetArrayFromImage(sitk_image)

    if return_sitk:
        return data, sitk_image, metadata

    return data, metadata
