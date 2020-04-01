"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
from fexp.utils.bbox import crop_to_bbox, BoundingBox


class CropAroundBbox:
    def __init__(self, output_size=None):
        self.output_size = np.asarray(output_size)

    def __call__(self, sample):
        bbox = BoundingBox(sample['bbox'])

        effective_output_size = self.output_size[-bbox.ndim:]
        new_bbox = bbox.bounding_box_around_center(effective_output_size).astype(int)

        sample['image'] = crop_to_bbox(sample['image'], new_bbox.squeeze(0))
        sample['mask'] = crop_to_bbox(sample['mask'], new_bbox)
        del sample['bbox']
        return sample


class RandomShiftBbox:
    def __init__(self, max_shift=None):
        self.max_shift = np.asarray(max_shift)

    def __call__(self, sample):
        # TODO: fexp, if is already BoundingBox, casting is not needed.
        bbox = BoundingBox(sample['bbox'])
        shift = np.random.randint(-self.max_shift, self.max_shift)
        new_bbox = (bbox + shift).astype(np.int)
        sample['bbox'] = new_bbox
        return sample


class RandomFlipTransform:
    def __init__(self, probability):
        self.mask = True
        self.axis = -1
        self.probability = probability

    def __call__(self, sample):
        if not np.random.random_sample() < self.probability:
            return sample

        sample['image'] = np.flip(sample['image'], axis=self.axis)
        if 'mask' in sample:
            sample['mask'] = np.flip(sample['mask'], axis=self.axis)

        return sample


class RandomGammaTransform:
    def __init__(self, gamma_range=(0.5, 1.5)):
        """
        Apply gamma transform.

        Beware, inputs need to be in (0, 1). This is not checked in the code.

        Parameters
        ----------
        gamma_range : tuple
        """

        self.gamma_range = gamma_range
        self.mask = False

    def __call__(self, sample):
        gamma = np.random.uniform(*self.gamma_range)
        sample['image'] = np.power(sample['image'], gamma)
        return sample


class RandomGaussianNoise:
    def __init__(self, std_dev=0.05, as_percentage=True):
        self.std_dev = std_dev
        self.as_percentage = as_percentage

    def __call__(self, sample):
        image = sample['image']
        if self.as_percentage:
            std_dev = (image.max() - image.min()) * self.std_dev
        else:
            std_dev = self.std_dev

        sample['image'] = image + np.random.normal(scale=std_dev, size=image.shape)
        return sample


class RandomLUT:
    """
    Applies a random lookup table or window level for mammograms.

    When a SIGMOID is set as VOILUTFunction, we can select a random window level and no LUT.
    """

    def __init__(self, pick_sensible=False, window_jitter_percentage=(0.05, 0.05)):
        self.pick_sensible = pick_sensible
        self.window_jitter_percentage = window_jitter_percentage

    def __call__(self, sample):
        mammogram = sample['mammogram']
        del sample['mammogram']
        num_dicom_luts = mammogram.num_dicom_luts
        num_center_widths = mammogram.num_center_widths
        voi_lut_function = mammogram.voi_lut_function

        if num_dicom_luts > 0 and voi_lut_function != 'SIGMOID':
            if self.pick_sensible:
                mammogram.set_lut(0)
            else:
                random_lut_idx = np.random.choice(num_dicom_luts)
                mammogram.set_lut(random_lut_idx)
        else:
            mammogram.set_lut(None)
            if not self.pick_sensible:
                if voi_lut_function == 'SIGMOID':
                    dicom_cw_idx = 0
                else:
                    dicom_cw_idx = np.random.choice(num_center_widths)
                dicom_window = [mammogram.dicom_window_center[dicom_cw_idx], mammogram.dicom_window_width[dicom_cw_idx]]
                dicom_window += np.random.uniform(size=2) * self.window_jitter_percentage
                mammogram.set_center_width(*dicom_window)
            else:
                dicom_window = [mammogram.dicom_window_center[0], mammogram.dicom_window_width[0]]
                mammogram.set_center_width(*dicom_window)

        sample['image'] = mammogram.image

        return sample
