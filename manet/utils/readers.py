# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import SimpleITK as sitk
import numpy as np
import tempfile
import pathlib

_DICOM_MODALITY_TAG = '0008|0060'
_DICOM_VOI_LUT_FUNCTION = '0028|1056'
_DICOM_WINDOW_CENTER_TAG = '0028|1050'
_DICOM_WINDOW_WIDTH_TAG = '0028|1051'
_DICOM_WINDOW_CENTER_WIDTH_EXPLANATION_TAG = '0028|1055'
_DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP = '0018|7034'
_DICOM_PATIENT_ORIENTATION = '0020|0020'
_DICOM_LATERALITY = '0020|0060'
_DICOM_IMAGE_LATERALITY = '0020|0062'
_DICOM_PHOTOMETRIC_INTERPRETATION = '0028|0004'

# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def read_image_as_sitk_image(filename):
    """
    Read file as a SimpleITK image trying to parse the error.

    Parameters
    ----------
    filename : Path or str

    Returns
    -------
    SimpleITK image.
    """
    try:
        sitk_image = sitk.ReadImage(str(filename))
    except RuntimeError as error:
        if 'itk::ERROR' in str(error):
            error = str(error).split('itk::ERROR')[-1]

        raise RuntimeError(error)

    return sitk_image


def read_image(filename, dtype=None, no_metadata=False, **kwargs):
    """Read medical image

    Parameters
    ----------
    filename : Path, str
        Path to image, can be any SimpleITK supported filename
    dtype : dtype
        The requested dtype the output should be cast.
    no_metadata : bool
        Do not output metadata

    Returns
    -------
    Image as ndarray and dictionary with metadata.
    """
    filename = pathlib.Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'{filename} does not exist.')

    new_spacing = kwargs.get('spacing', False)
    if new_spacing and np.all(np.asarray(new_spacing) <= 0):
        new_spacing = False

    metadata = {}
    sitk_image = read_image_as_sitk_image(filename)

    # TODO: A more elaborate check for dicom can be needed, not necessarly all dicom files have .dcm as extension.
    if filename.suffix.lower() == '.dcm' and kwargs.get('dicom_keys', None):
        dicom_data = {}
        metadata_keys = sitk_image.GetMetaDataKeys()
        for v in kwargs['dicom_keys']:
            dicom_data[v] = None if v not in metadata_keys else sitk_image.GetMetaData(v).strip()
        metadata['dicom_tags'] = dicom_data

    orig_shape = sitk.GetArrayFromImage(sitk_image).shape
    if new_spacing:
        sitk_image, orig_spacing = resample_sitk_image(
            sitk_image,
            spacing=new_spacing,
            interpolator=kwargs.get('interpolator', None),
            fill_value=0
        )
        metadata.update(
            {'orig_spacing': tuple(orig_spacing), 'orig_shape': orig_shape})

    image = sitk.GetArrayFromImage(sitk_image)

    metadata.update({
        'filename': filename.resolve(),
        'depth': sitk_image.GetDepth(),
        'spacing': sitk_image.GetSpacing(),
        'origin': sitk_image.GetOrigin(),
        'direction': sitk_image.GetDirection()
    })

    if dtype:
        image = image.astype(dtype)

    if no_metadata:
        return image

    return image, metadata


def read_mammogram(filename, dtype=np.int):
    """
    Read mammograms in dicom format. Dicom tags which:
    - flip images horizontally,
    - VOI Lut Function before displaying,

    are read and set appropriately.

    Parameters
    ----------
    filename : pathlib.Path or str
    dtype : dtype

    Returns
    -------
    np.ndarray, dict

    TODO: Monochrome 2
    """
    extra_tags = [_DICOM_MODALITY_TAG, _DICOM_VOI_LUT_FUNCTION,
                  _DICOM_LATERALITY, _DICOM_IMAGE_LATERALITY,
                  _DICOM_WINDOW_CENTER_TAG, _DICOM_WINDOW_CENTER_TAG,
                  _DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP, _DICOM_PATIENT_ORIENTATION,
                  _DICOM_PHOTOMETRIC_INTERPRETATION]

    image, metadata = read_image(filename, dicom_keys=extra_tags, dtype=dtype)
    dicom_tags = metadata['dicom_tags']

    modality = dicom_tags[_DICOM_MODALITY_TAG]
    if not modality == 'MG':
        raise ValueError(f'{filename} is not a mammogram. Wrong Modality in DICOM header.')
    if not metadata['depth'] == 1:
        raise ValueError(f'First dimension of mammogram should be one.')

    # Remove the depth dimension
    image = image.reshape(list(image.shape)[1:])

    # Sometimes a function, the VOILUTFunction, needs to be applied before displaying the mammogram.
    voi_lut_function = dicom_tags[_DICOM_VOI_LUT_FUNCTION] if dicom_tags[_DICOM_VOI_LUT_FUNCTION] else 'LINEAR'
    if voi_lut_function == 'LINEAR':
        pass
    elif voi_lut_function == 'SIGMOID':
        # https://dicom.innolitics.com/ciods/nm-image/voi-lut/00281056
        image_min = image.min()
        image_max = image.max()
        window_center = float(dicom_tags[_DICOM_WINDOW_CENTER_TAG])
        window_width = float(dicom_tags[_DICOM_WINDOW_WIDTH_TAG])
        image = (image_max - image_min) / (1 + np.exp(-4 * (image - window_center) / window_width)) + image_min
        if dtype:
            image = image.astype(dtype)
    else:
        raise NotImplementedError(f'VOI LUT Function {voi_lut_function} is not implemented.')

    # Photometric Interpretation determines how to read the pixel values and if they should be inverted
    photometric_interpretation = dicom_tags[_DICOM_PHOTOMETRIC_INTERPRETATION]
    if photometric_interpretation == 'MONOCHROME2':
        pass
    else:
        raise NotImplementedError(f'Photometric Interpretation {photometric_interpretation} is not implemented.')

    laterality = dicom_tags[_DICOM_LATERALITY] or dicom_tags[_DICOM_IMAGE_LATERALITY]
    metadata['laterality'] = laterality

    # Sometimes a horizontal flip is required:
    # https://groups.google.com/forum/#!msg/comp.protocols.dicom/X4ddGYiQOzs/g04EDChOQBwJ
    needs_horizontal_flip = dicom_tags[_DICOM_FIELD_OF_VIEW_HORIZONTAL_FLIP] == 'YES'
    if laterality:
        # Check patient position
        orientation = dicom_tags[_DICOM_PATIENT_ORIENTATION].split('\\')[0]
        if (laterality == 'L' and orientation == 'P') or (laterality == 'R' and orientation == 'A'):
            needs_horizontal_flip = True

    if needs_horizontal_flip:
        image = np.ascontiguousarray(np.fliplr(image))

    del metadata['dicom_tags']
    del metadata['depth']
    del metadata['direction']
    del metadata['origin']

    metadata['spacing'] = metadata['spacing'][:-1]

    return image, metadata


def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.

    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()
    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) if s else orig_spacing[idx] for idx, s in enumerate(spacing)]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    # SimpleITK expects lists
    new_size = [int(s) if spacing[idx] else int(orig_size[idx]) for idx, s in enumerate(new_size)]

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(
        sitk_image,
        new_size,
        sitk.Transform(),
        sitk_interpolator,
        orig_origin,
        new_spacing,
        orig_direction,
        fill_value,
        orig_pixelid
    )

    return resampled_sitk_image, orig_spacing


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
