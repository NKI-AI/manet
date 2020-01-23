# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import cv2
from skimage.transform import rescale, rotate
from manet.utils.bbox import crop_to_bbox, combine_bbox, BoundingBox
from manet.utils.image import clip_and_scale
from config.base_config import cfg


# TODO: fexp
class Compose(object):
    """Compose several transforms together. For instance, normalization combined with a flip
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        repr_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            repr_string += '\n'
            repr_string += f'    {transform}'
        repr_string += '\n)'
        return repr_string


# TODO: fexp
class ClipAndScale(object):
    """Clip input array and rescale image data.
    """

    def __init__(self, clip_range, source_interval, target_interval):
        self.clip_range = clip_range
        self.source_interval = source_interval
        self.target_interval = target_interval

    def apply_transform(self, data):
        return clip_and_scale(data, self.clip_range, self.source_interval, self.target_interval)

    def __call__(self, sample):
        sample['image'] = self.apply_transform(sample['image'])
        return sample


class RandomRotation(object):
    def __init__(self, **kwargs):
        std = kwargs.get('std', 5.0)
        self.mask = True
        self.angle = -std + np.random.random_sample()*2.0*std
        self.axis = kwargs.get('axis', 0)

    def apply(self, x):
        if np.abs(self.angle) < 1.0:
            return x
        is_mask = x.dtype in [np.uint8, np.uint16]
        amax = np.max(np.abs(x)) + 1.0
        if len(x.shape) == 2:
            if not is_mask:
                return rotate(x / amax, self.angle, order=1)*amax
            else:
                return np.rint(rotate(x.astype(np.float32)/255.0, self.angle, order=1)*255.0).astype(x.dtype)
        else:
            if self.axis == 0:
                if not is_mask:
                    slices = [rotate(x[idx, :, :] / amax, self.angle, order=1)*amax for idx in range(x.shape[0])]
                else:
                    slices = [np.rint(rotate(x[idx, :, :].astype(np.float32)/255.0, self.angle, order=1)*255.0)
                              for idx in range(x.shape[0])]
                return np.stack(slices, axis=0).astype(x.dtype)
            elif self.axis == 1:
                if not is_mask:
                    slices = [rotate(x[:, idx, :] / amax, self.angle, order=1)*amax for idx in range(x.shape[1])]
                else:
                    slices = [np.rint(rotate(x[:, idx, :].astype(np.float32)/255.0, self.angle, order=1)*255.0)
                              for idx in range(x.shape[1])]
                return np.stack(slices, axis=1).astype(x.dtype)
            else:
                if not is_mask:
                    slices = [rotate(x[:, :, idx] / amax, self.angle, order=1)*amax for idx in range(x.shape[2])]
                else:
                    slices = [np.rint(rotate(x[:, :, idx].astype(np.float32)/255.0, self.angle, order=1)*255.0)
                              for idx in range(x.shape[2])]
                return np.stack(slices, axis=2).astype(x.dtype)


class CropAroundBbox(object):
    def __init__(self, output_size=None):
        self.output_size = np.asarray(output_size)

    def __call__(self, sample):
        bbox = BoundingBox(sample['bbox'])

        effective_output_size = self.output_size[-bbox.ndim:]
        if np.all(bbox.size <= effective_output_size):
            # A center crop is fine.
            new_bbox = bbox.bounding_box_around_center(effective_output_size).astype(int)
        else:
            starting_point = bbox.coordinates
            delta = np.clip(effective_output_size - bbox.size, 0, bbox.size.max()) // 2
            jitter = np.random.randint(-delta, delta + 1)
            # Here it makes sense to overwrite the add operator of bounding box
            new_bbox = BoundingBox(combine_bbox(starting_point - jitter, effective_output_size), dtype=int)

        # del sample['bbox']
        # TODO: Extra dimension is not always needed.

        try:
            sample['image'] = crop_to_bbox(sample['image'], new_bbox.squeeze(0))
        except ValueError as e:
            print(new_bbox.squeeze(0), new_bbox, sample['image'].shape)
            raise ValueError(e)

        sample['mask'] = crop_to_bbox(sample['mask'], new_bbox)

        return sample


class RandomGammaTransform(object):
    def __init__(self, **kwargs):
        rng = kwargs.get('rng', cfg.UNET.GAMMA_RANGE)
        self.alpha = rng * np.random.random_sample() + 1.0 - 0.5*rng
        self.mask = False

    def apply(self, sample):
        sample['image'] = np.power(sample['image'], self.alpha)
        return sample


class RandomGaussianNoise(object):
    def __init__(self, **kwargs):
        self.std = kwargs.get('std', cfg.UNET.NOISE_STD)
        self.mask = False

    def apply(self, x):
        return x * (1.0 + np.random.randn(*x.shape).astype(np.float32) * self.std)


class RandomShiftTransform(object):
    def __init__(self, **kwargs):
        self.mask = True
        delta = kwargs.get('delta', 32)
        dim = kwargs.get('dim', 3)
        ignore_dim = kwargs.get('ignore_dim', 0)
        self.delta = np.random.randint(1 - delta, high=delta, size=dim)
        if ignore_dim is not None:
            self.delta[ignore_dim] = 0

    def apply(self, x):
        bbox = combine_bbox(self.delta, x.shape)
        return crop_to_bbox(x, bbox)


class RandomZoomTransform(object):
    def __init__(self, **kwargs):
        self.mask = True
        alpha = kwargs.get('alpha', 0.2)
        self.delta = alpha * np.random.random_sample() + 1.0 - 0.5 * alpha

    def apply(self, x):
        in_shape = np.array(x.shape)
        amax = np.max(np.abs(x)) + 1.0
        if x.dtype == np.uint16 or x.dtype == np.uint8:
            _y = rescale(x.astype(np.float32) / 255.0, self.delta, order=1) * 255.0
            y = np.rint(_y).astype(x.dtype)
        else:
            y = rescale(x / amax, self.delta)*amax
        t_shape = np.array(y.shape)
        delta = (t_shape - in_shape) // 2
        if self.delta > 1:
            bbox = combine_bbox(delta, in_shape)
        else:
            bbox = combine_bbox(-delta, in_shape)
        return crop_to_bbox(x, bbox)


class RandomElasticTransform(object):
    def __init__(self, **kwargs):
        """
        Elastic deformation in 2D or 3D per slice. For 3D volumes (z, x, y) keeps z axis unperturbed.
        """
        assert "in_shape" in kwargs, "Shape should be available to the constructor"
        self.mask = True
        in_shape = kwargs['in_shape']
        std = kwargs.get('std', 0.6)
        sigma = kwargs.get('sigma', 0)
        scale = kwargs.get('scale', 0.5)
        blur_size = int(std * np.max(in_shape)) | 1
        self.rand_x = scale * cv2.GaussianBlur(
            (np.random.uniform(size=in_shape[-2:]) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * 2 * in_shape[1 if len(in_shape) == 3 else 0]
        self.rand_y = scale * cv2.GaussianBlur(
            (np.random.uniform(size=in_shape[-2:]) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * 2 * in_shape[2 if len(in_shape) == 3 else 1]

    def apply(self, x):
        if len(x.shape) == 2:
            grid_x, grid_y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        else:
            grid_x, grid_y = np.meshgrid(np.arange(x.shape[2]), np.arange(x.shape[1]))
        grid_x = (grid_x + self.rand_x).astype(np.float32)
        grid_y = (grid_y + self.rand_y).astype(np.float32)

        is_mask = x.dtype in [np.uint16, np.uint8]
        if len(x.shape) == 2:
            if not is_mask:
                return cv2.remap(x, grid_x, grid_y,
                                 borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR).astype(x.dtype)
            else:
                t = cv2.remap(x.astype(np.float32) / 255.0, grid_x, grid_y,
                              borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)*255.0
                return np.rint(t).astype(x.dtype)
        else:
            if not is_mask:
                out_stack = [cv2.remap(x[i, :, :], grid_x, grid_y,
                                       borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR) for i in
                             range(x.shape[0])]
            else:
                out_stack = [cv2.remap(x[i, :, :].astype(np.float32) / 255.0, grid_x, grid_y,
                                       borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)*255.0 for i in
                             range(x.shape[0])]
            return np.stack(np.rint(out_stack), axis=0).astype(x.dtype)


class RandomFlipTransform(object):
    def __init__(self):
        self.mask = True
        self.axis = -1
        self.do_flip = (np.random.random_sample() < 0.5)

    def apply(self, x):
        if self.do_flip:
            return np.flip(x, axis=self.axis)
        else:
            return x


class RandomFlipTransformExt(object):
    def __init__(self):
        idx = np.random.choice(len(cfg.UNET.FLIP_AXIS), 1)[0]
        self.mask = cfg.UNET.FLIP_MASK[idx]
        self.axis = cfg.UNET.FLIP_AXIS[idx]
        self.do_flip = np.random.random_sample() < 0.5

    def apply(self, x):
        if self.do_flip:
            return np.flip(x, axis=self.axis)
        else:
            return x
