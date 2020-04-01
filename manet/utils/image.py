"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
import numpy as np
import pydicom
from manet.utils.dicom import DICOM_WINDOW_CENTER, DICOM_WINDOW_WIDTH, DICOM_WINDOW_CENTER_WIDTH_EXPLANATION, \
    build_dicom_lut
from fexp.image import clip_and_scale


class Image:
    """
    Rudimentary object to allow for storing image properties and ability to write to file.

    Do not trust on this! API can change.
    """
    def __init__(self, data, data_origin=None, header=None, *args, **kwargs):
        self.data = data
        self.data_origin = data_origin
        self.header = header
        self.spacing = None if not header else header.get('spacing', None)

    @staticmethod
    def shape(self):
        return self.data.shape

    def to_filename(self, filename, compression=True):
        manet.save(self, filename, compression=compression)


class MammogramImage(Image):
    def __init__(self, data, image_fn, header, voi_lut_function=None, view=None, laterality=None):
        super().__init__(data, image_fn, header)

        self.raw_image = data
        self.header = header

        self.voi_lut_function = voi_lut_function
        if self.voi_lut_function not in ['LINEAR', 'LINEAR_EXACT', 'SIGMOID']:
            raise ValueError(
                f'VOI LUT Function {self.voi_lut_function} is not supported by the DICOM standard.')

        self.view = view
        self.laterality = laterality

        self._image = None

        # Window leveling
        self._output_range = (0.0, 1.0)
        self._current_set_center_width = [None, None]
        self.num_dicom_center_widths = 0
        self._parse_window_level()

        # LUTs
        self._uniques = None
        self._current_set_lut = None
        self.dicom_luts = []
        self.num_dicom_luts = 0
        self._parse_luts()

    def _parse_window_level(self):
        window_center = self.header['dicom_tags'][DICOM_WINDOW_CENTER]
        window_width = self.header['dicom_tags'][DICOM_WINDOW_WIDTH]
        explanation = self.header['dicom_tags'][DICOM_WINDOW_CENTER_WIDTH_EXPLANATION]

        if window_center and window_width:
            self.dicom_window_center = [float(_) for _ in window_center.split('\\')]
            self.dicom_window_width = [float(_) for _ in window_width.split('\\')]
            if not len(self.dicom_window_width) == len(self.dicom_window_center):
                raise ValueError(f'Number of widths and center mismatch.')
            self.num_dicom_center_widths = len(self.dicom_window_width)

        if explanation:
            self.dicom_center_width_explanation = [_.strip() for _ in explanation.split('\\')]

        if self.voi_lut_function == 'SIGMOID':
            # In this case, window and center always need to be set.
            if not (len(self.dicom_window_center) == 1 and len(self.dicom_window_width) == 1):
                raise ValueError(f'If VOILUTFunction is set to `SIGMOID`, '
                                 f'tags {DICOM_WINDOW_CENTER} and {DICOM_WINDOW_WIDTH} need to be set with one value.')
            self._current_set_center_width = [self.dicom_window_center[0], self.dicom_window_width[0]]

    def _parse_luts(self):
        # SimpleITK does not yet support sequence tags, therefore read with pydicom.
        dcm = pydicom.read_file(str(self.data_origin), stop_before_pixels=True)
        if not self._uniques:
            self._uniques = np.unique(self.raw_image)
        voi_lut_sequence = getattr(dcm, 'VOILUTSequence', [])

        for voi_lut in voi_lut_sequence:
            self.num_dicom_luts += 1
            lut_descriptor = list(voi_lut.LUTDescriptor)
            lut_explanation = voi_lut.LUTExplanation
            lut_data = list(voi_lut.LUTData)
            len_lut = lut_descriptor[0] if not lut_descriptor[0] == 0 else 2 ** 16
            first_value = lut_descriptor[1]  # TODO: This assumes that mammograms are always unsigned integers.
            # number_of_bits_lut_data = lut_descriptor[2]

            self.dicom_luts.append((lut_explanation, lut_data, len_lut, first_value))

    def set_lut(self, idx):
        if idx is not None and (idx < 0 or idx >= len(self.dicom_luts)):
            raise ValueError(f'Incorrect LUT index. Got {idx}.')
        self._current_set_lut = idx

    def set_center_width(self, window_center, window_width):
        if window_width <= 0:
            raise ValueError(f'window width should be larger than 0. Got {window_width}.')
        if not window_center or not window_width:
            raise ValueError(f'center and width should both be set.')

        self._current_set_center_width = [window_center, window_width]

    def _apply_sigmoid(self, image, window_center, window_width):
        # https://dicom.innolitics.com/ciods/nm-image/voi-lut/00281056
        image_min = image.min()
        image_max = image.max()
        output = (image_max - image_min) / (1 + np.exp(-4 * (image - window_center) / window_width)) + image_min
        return output.astype(image.dtype)

    def _apply_linear_exact(self, image, window_center, window_width):
        output = np.clip(image,
                         window_center - window_width / 2,
                         window_center + window_width / 2)

        output = ((output - window_center) / window_width) + 0.5
        output = output * (self._output_range[1] - self._output_range[0]) + self._output_range[0]
        return output

    def _apply_linear(self, image, window_center, window_width):
        output = np.clip(image,
                         window_center - 0.5 - (window_width - 1) / 2,
                         window_center - 0.5 + (window_width - 1) / 2)

        output = ((output - (window_center - 0.5)) / (window_width - 1)) + 0.5
        output = output * (self._output_range[1] - self._output_range[0]) + self._output_range[0]
        return output

    @property
    def image(self):
        if self._current_set_lut is not None and self._current_set_center_width is not None:
            warnings.warn(f'Both LUT and center width are set, this can lead to unexpected results.')

        if self._current_set_lut is not None:
            _, lut_data, len_lut, first_value = self.dicom_luts[self._current_set_lut]
            LUT = build_dicom_lut(self._uniques, lut_data, len_lut, first_value)
            self._image = clip_and_scale(LUT[self.raw_image], None, None, self._output_range)
        else:
            self._image = clip_and_scale(self.raw_image, None, None, self._output_range)

        if all(self._current_set_center_width):
            if self.voi_lut_function == 'LINEAR':
                self._image = self._apply_linear(self._image, *self._current_set_center_width)
            elif self.voi_lut_function == 'LINEAR_EXACT':
                self._image = self._apply_linear_exact(self._image, *self._current_set_center_width)
            elif self.voi_lut_function == 'SIGMOID':
                self._image = clip_and_scale(
                    self._apply_sigmoid(self._image, *self._current_set_center_width), None, None, self._output_range)
            else:
                raise ValueError(f'VOI LUT Function {self.voi_lut_function} is not supported by the DICOM standard.')

        return self._image

    def to_filename(self, *args, **kwargs):
        raise NotImplementedError(f'API unstable. Saving the raw image will create difficulties when parsing LUTs.')
