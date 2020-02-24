"""
Copyright (c) Nikita Moriakov and Jonas Teuwen


This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import SimpleITK as sitk
import tempfile

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
