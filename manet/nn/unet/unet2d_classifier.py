"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# encoding: utf-8
__author__ = 'Jonas Teuwen, Sil van de Leemput, Nikita Moriakov'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class GradMul(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return self.lambd * grad_output


def grad_mul(x, lambd):
    return GradMul(lambd)(x)


def unet_shape_out(input_shape, depth, ignore_dim=None):
    """
    Given the input shape of a u-net and the depth, compute the output shape (receptive field)

    Parameters
    ----------
    input_shape : list
    depth : list

    Returns
    -------
    list

    """
    input_shape = np.asarray(input_shape)
    assert np.all((input_shape + 4) / 2 ** depth % 1 == 0), 'input shape + 4 has to be divisible by 2 ** depth'
    down_shape = (input_shape + 4) / 2 ** depth - 8
    up_shape = 2 ** depth * (down_shape - 4) + 4
    if ignore_dim is not None:
        for d in ignore_dim:
            up_shape[d] = input_shape[d]
    return up_shape.astype(np.int)


def unet_shape_in(output_shape, depth, ignore_dim=None):
    """
    Given the desired receptive field and depth, give the required input shape.

    Parameters
    ----------
    output_shape : list
    depth : list

    Returns
    -------
    list

    """
    output_shape = np.asarray(output_shape)
    assert np.all((output_shape - 4) / 2 ** depth % 1 == 0), 'output shape - 4 has to be divisible by 2 ** depth'
    down_shape = (output_shape - 4) / 2 ** depth + 4
    input_shape = 2 ** depth * (down_shape + 8) - 4
    if ignore_dim is not None:
        for d in ignore_dim:
            input_shape[d] = output_shape[d]
    return input_shape.astype(np.int)


def unet_admissible_outputs(largest_output, depth):
    # TODO write function
    pass


class UNet(nn.Module):
    def __init__(self, num_channels, num_classes, valid=True,
                 upsample_mode='nearest', depth=4, dropout_depth=2, dropout_prob=0.5, channels_base=64,
                 bn_conv_order='brcbrc', domain_classifier=False, forward_domain_cls=False, num_domains=1,
                 valid_mode=False, gradient_multiplication_scale=False):
        """2D U-Net model implementation 4 x down and upscale

        Parameters
        ----------
            num_channels : int
                number of input channels
            num_classes : int
                number of output classes
            valid : bool
                use valid convolutions instead of zero padded convolutions
            upsample_mode : str
                upsampling mode, can be: nearest, bilinear, etc...
            channels_base : int
                initial base channels for each double convolution
            bn_conv_order : str
                order of batch normalization, ReLU and convolution. Can be either 'brcbrc' or 'cbrcbr'.

        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.inc = Inconv(num_channels, channels_base, valid, bn_conv_order=bn_conv_order)
        self.depth = depth
        self.use_classifier = domain_classifier
        self.forward_domain_cls = forward_domain_cls
        self.num_domains = num_domains
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.clsf = None

        self.gradient_multiplication_scale = gradient_multiplication_scale

        self.valid_mode = valid_mode

        for idx in range(self.depth):
            dropout = dropout_prob if idx > self.depth - dropout_depth - 1 else False
            downer = Down(channels_base * 2 ** idx, channels_base * 2 ** (idx + 1), valid, dropout=dropout,
                          bn_conv_order=bn_conv_order)
            self.downs.append(downer)
            if self.forward_domain_cls and self.use_classifier:
                upper = UpX(channels_base * 2 ** (depth - idx) + 1, channels_base * 2 ** (depth - idx - 1),
                            channels_base * 2 ** (depth - idx - 1), valid, mode=upsample_mode,
                            bn_conv_order=bn_conv_order)
            else:
                upper = Up(channels_base * 2 ** (depth - idx), channels_base * 2 ** (depth - idx - 1),
                           channels_base * 2 ** (depth - idx - 1), valid, mode=upsample_mode,
                           bn_conv_order=bn_conv_order)
            self.ups.append(upper)

        self.outc = Outconv(channels_base, num_classes)
        if self.forward_domain_cls:
            self.outc_softmax = torch.nn.Softmax(dim=1)

        if self.use_classifier:
                self.clsf = Classifier(channels_base * 2 ** depth, num_domains=self.num_domains,
                                       grad_scale=1.0 if not self.gradient_multiplication_scale else
                                       self.gradient_multiplication_scale)
        self.init_weights()

    def forward(self, x):
        xs = [self.inc(x)]
        for idx in range(self.depth):
            xs.append(self.downs[idx](xs[idx]))

        if self.use_classifier:
            y = self.clsf(xs[-1])[:, :, 0, 0]
            if self.forward_domain_cls:
                z = self.outc_softmax(y.detach())[:, 0].view(-1, 1, 1, 1)

        x = self.ups[0](xs[-1], xs[-2]) if not (self.use_classifier and self.forward_domain_cls) else self.ups[0](
            xs[-1], xs[-2], z)
        for idx in range(self.depth - 1):
            x = self.ups[idx + 1](x, xs[self.depth - idx - 2]) if not (
                        self.use_classifier and self.forward_domain_cls) else self.ups[idx + 1](x, xs[
                self.depth - idx - 2], z)

        x = self.outc(x)
        if self.use_classifier:
            return x, y
        else:
            return x

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.reset_parameters()


def number_groups(x):
    if x % 8 == 0:
        print("{}->{}".format(x, 8))
        return 8
    for i in range(2, int(np.sqrt(x))):
        if x % i == 0:
            print("{}->{}".format(x, i))
            return i
    print("{}->{}".format(x, x))
    return x


class Double_conv_old(nn.Module):
    def __init__(self, in_ch, out_ch, valid=True, bn_conv_order='brcbrc', stride=1, norm='batch_norm'):
        super(Double_conv, self).__init__()

        use_batch_norm = norm == 'batch_norm'
        group_norm_num_groups = int(norm.split('_')[-1]) if norm.startswith('group_norm') else None

        if bn_conv_order == 'cbrcbr':
            if norm:
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True)
                )
            else:
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    nn.ReLU(inplace=True)
                )

        elif bn_conv_order == 'brcbrc':
            if norm:
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    (nn.BatchNorm2d(in_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, in_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1))
                )

            else:
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1))
                )

        else:
            raise ValueError(f'bn_conv_order:"{bn_conv_order}" not supported.')

    def forward(self, x):
        x = self.conv(x)
        return x


class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, valid=True, bn_conv_order='brcbrc', stride=1, norm='batch_norm'):
        super(Double_conv, self).__init__()

        use_batch_norm = norm == 'batch_norm'
        group_norm_num_groups = int(norm.split('_')[-1]) if norm.startswith('group_norm') else None

        if norm:
            if bn_conv_order == 'cbrcbr':
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True)
                )
            elif bn_conv_order == 'brcbrc':
                self.num_gr = 8 if out_ch % 8 == 0 else out_ch
                self.conv = nn.Sequential(
                    (nn.BatchNorm2d(in_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, in_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                    (nn.BatchNorm2d(out_ch)) if use_batch_norm else (nn.GroupNorm(group_norm_num_groups, out_ch)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1))
                )
            else:
                raise ValueError(f'bn_conv_order:"{bn_conv_order}" not supported.')

        else:
            self.num_gr = 8 if out_ch % 8 == 0 else out_ch
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=(0 if valid else 1)),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, valid=True, bn_conv_order='brcbrc'):
        super(Inconv, self).__init__()
        self.out_ch = out_ch
        self.conv = Double_conv(in_ch, out_ch, valid=valid, bn_conv_order=bn_conv_order)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, valid=True, dropout=0.0, bn_conv_order='brcbrc', ):
        super(Down, self).__init__()
        self.out_ch = out_ch
        self.mp = nn.MaxPool2d(2)
        self.doubleconv = Double_conv(in_ch, out_ch, valid=valid, bn_conv_order=bn_conv_order)
        self.dropout = None
        if dropout and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.mp(x)
        x = self.doubleconv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, join_ch, out_ch, valid=True, mode='bilinear', bn_conv_order='brcbrc'):
        super(Up, self).__init__()
        self.mode = mode
        self.valid = valid
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.doubleconv = Double_conv(in_ch + join_ch, out_ch, valid=valid, bn_conv_order=bn_conv_order)
        self.out_ch = out_ch

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        t1 = diffX // 2
        t2 = diffX - t1
        s1 = diffY // 2
        s2 = diffY - s1
        x2 = F.pad(x2, (s1, s2, t1, t2))
        x = torch.cat([x2, x1], dim=1)
        x = self.doubleconv(x)
        return x


class UpX(nn.Module):
    def __init__(self, in_ch, join_ch, out_ch, valid=True, mode='bilinear', bn_conv_order='brcbrc'):
        super(UpX, self).__init__()
        self.mode = mode
        self.valid = valid
        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.doubleconv = Double_conv(in_ch + join_ch, out_ch, valid=valid, bn_conv_order=bn_conv_order)
        self.out_ch = out_ch

    def forward(self, x1, x2, cls):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        t1 = diffX // 2
        t2 = diffX - t1
        s1 = diffY // 2
        s2 = diffY - s1
        x2 = F.pad(x2, (s1, s2, t1, t2))
        shape = list(x1.size())
        x3 = cls * torch.ones((shape[0], 1, shape[2], shape[3])).cuda()
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.doubleconv(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2, valid_mode=True):
        super(Outconv, self).__init__()
        if dim == 2:
            self.conv = nn.Conv2d(in_ch, out_ch, 1, padding=(0 if not valid_mode else 1))
        else:
            self.conv = nn.Conv3d(in_ch, out_ch, 1, padding=(0 if not valid_mode else 1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_ch, num_domains=1, valid_mode=True, bn_conv_order='brcbrc', grad_scale=0.5):
        super(Classifier, self).__init__()

        self.grad_scale = grad_scale
        self.doubleconv = Double_conv(in_ch, in_ch, valid=valid_mode, bn_conv_order=bn_conv_order, stride=2)
        self.outconv = nn.Conv2d(in_ch, num_domains, 1)

    def forward(self, x):
        x = self.doubleconv(grad_mul(x, self.grad_scale))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.outconv(x)

        return x

