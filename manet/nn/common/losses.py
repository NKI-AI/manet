# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn.functional as F
import logging
import numpy as np
from math import exp


def create_kernel(kernel_size, in_ch, sigma=1.5, dim=2):
    # Create convolutional kernel with Gaussian PDF
    # Output is tensor of shape (1, in_ch, kernel_size, kernel_size)
    gaussian = torch.Tensor([exp(-(i - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for i in range(kernel_size)])
    gaussian = gaussian / gaussian.sum()
    _1D_kernel = gaussian.unsqueeze(1)
    if dim == 2:
        _2D_kernel = _1D_kernel.mm(_1D_kernel.t()).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.Tensor(_2D_kernel.expand(in_ch, 1, kernel_size, kernel_size).contiguous())
    else:
        _2D_kernel = _1D_kernel.mm(_1D_kernel.t()).float().unsqueeze(-1)
        _3D_kernel = torch.matmul(_2D_kernel, _1D_kernel.t()).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.Tensor(_3D_kernel.expand(in_ch, 1, kernel_size, kernel_size, kernel_size).contiguous())
    return kernel


def _ssim(img1, img2, kernel, kernel_size, in_ch, batch_average=True):
    if len(kernel.size()) == 4:
        mu1 = F.conv2d(img1, kernel, padding=kernel_size // 2, groups=in_ch)
        mu2 = F.conv2d(img2, kernel, padding=kernel_size // 2, groups=in_ch)
    else:
        mu1 = F.conv3d(img1, kernel, padding=kernel_size // 2, groups=in_ch)
        mu2 = F.conv3d(img2, kernel, padding=kernel_size // 2, groups=in_ch)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    if len(kernel.size()) == 4:
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size // 2, groups=in_ch) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size // 2, groups=in_ch) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size // 2, groups=in_ch) - mu1_mu2
    else:
        sigma1_sq = F.conv3d(img1 * img1, kernel, padding=kernel_size // 2, groups=in_ch) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, kernel, padding=kernel_size // 2, groups=in_ch) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel_size // 2, groups=in_ch) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if batch_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, in_ch, kernel_size=11, reduction='none'):
        super(SSIM, self).__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.kernel_size = kernel_size
        assert reduction in ('mean', 'none')
        self.reduction = reduction
        self.in_ch = in_ch
        self.kernel = create_kernel(kernel_size, self.in_ch)
        self.logger.debug(f'Kernel size: {kernel_size}, number of input channels: {self.in_ch}')
        import sys
        sys.exit(self.kernel)

    def forward(self, img1, img2):
        if img1.is_cuda and not self.kernel.is_cuda:
            self.kernel = self.kernel.cuda(img1.get_device())
        # self.kernel = self.kernel.type_as(img1)
        if self.reduction == 'none':
            batch_average = False
        else:
            batch_average = True
        return _ssim(img1, img2, self.kernel, self.kernel_size, self.in_ch, batch_average)


def ssim(img1, img2, kernel_size=11, batch_average=True):
    in_ch = img1.size()[1]
    dim = 3 if len(img1.size()) == 5 else 2
    kernel = create_kernel(kernel_size, in_ch, dim=dim)

    if img1.is_cuda:
        kernel = kernel.cuda(img1.get_device())
    kernel = kernel.type_as(img1)

    return _ssim(img1, img2, kernel, kernel_size, in_ch, batch_average)


def psnr(x_result, x_true):
    """
    This function is a torch implementation of skimage.metrics.compare_psnr

    Parameters
    ----------
    x_result
    x_true

    Returns
    -------

    """
    batch_sz = x_true.size(0)
    true_view = x_true.view(batch_sz, -1)
    result_view = x_result.view(batch_sz, -1)
    maxval = torch.max(true_view, 1)[0]

    mse = torch.mean((true_view - result_view) ** 2, 1)
    psnrs = 20.0 * torch.log10(maxval) - 10.0 * torch.log10(mse)
    return psnrs.mean()


class TopkL1(torch.nn.Module):
    def __init__(self, k=0.1):
        super(TopkL1, self).__init__()
        self.k = k

    def forward(self, x, y):
        t_size = x.size()
        num = np.prod(t_size[1:])
        loss = F.l1_loss(x, y, reduction='none').view(t_size[0], -1)
        u, _ = torch.topk(loss, int(self.k * num))
        return torch.mean(u)


class HardDice(torch.nn.Module):
    def __init__(self, cls=0, weight=None, size_average=True, single_channel=False, binary_cls=False, reduce=True):
        super(HardDice, self).__init__()
        self.cls = cls
        self.reduce = reduce
        self.single_channel = single_channel
        self.binary_cls = binary_cls

    def forward(self, logits, targets):
        sizes = list(logits.size())
        eps = 0.00001
        one = torch.tensor(1.0).to(logits.device)
        zero = torch.tensor(0.0).to(logits.device)
        thr = (torch.tensor(0.5)).to(logits.device).type(logits.type())
        if self.binary_cls:
            m1 = torch.where(logits > thr, one, zero).view(sizes[0], -1)
        else:
            _, classes = torch.max(logits, dim=1)
            m1 = torch.where(classes == self.cls, one, zero).view(sizes[0], -1)
            
        if self.binary_cls:
            m2 = targets.view(sizes[0], -1)
        else:
            if not self.single_channel:
                if len(sizes) == 4:
                    m2 = targets[:, self.cls, :, :].view(sizes[0], -1)
                elif len(sizes) == 5:
                    m2 = targets[:, self.cls, :, :, :].view(sizes[0], -1)
                else:
                    raise NotImplementedError("Tensor shape {} not supported.".format(sizes))
            else:
                m2 = torch.where(targets == self.cls, one, zero).view(sizes[0], -1)

        intersection = m1 * m2
        score = (2. * intersection.sum(1).float() + eps) / (m1.sum(1).float() + m2.sum(1).float() + eps)

        if self.reduce:
            return score.sum() / sizes[0]
        else:
            return score


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, single_channel=False):
        super(SoftDiceLoss, self).__init__()
        self.single_channel = single_channel

    def forward(self, logits, targets):
        smooth = 1
        sizes = list(logits.size())
        num_samples = sizes[0]
        probs = F.softmax(logits, dim=1)  # Compute softmax in channel dimension instead of F.sigmoid(logits)
        m1 = probs.view(num_samples, -1)
        if not self.single_channel:
            m2 = targets.view(num_samples, -1)
        else:
            m2 = torch.zeros_like(logits)

            if len(sizes) == 4:
                m2.scatter_(1, targets.view(num_samples, 1, sizes[2], sizes[3]).long(), 1)
            elif len(sizes) == 5:
                m2.scatter_(1, targets.view(num_samples, 1, sizes[2], sizes[3], sizes[4]).long(), 1)
            else:
                raise NotImplementedError("Shape {} not supported!".format(sizes))

            m2 = m2.view(num_samples, -1)

        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)

        return 1 - score.sum() / num_samples


class TopkCrossEntropy(torch.nn.Module):
    def __init__(self, weight=None, top_k=0.5, reduce=True):
        super(TopkCrossEntropy, self).__init__()
        self.reduce = reduce
        if weight is None:
            self.loss = torch.nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.top_k = top_k

    def forward(self, input, target):
        sizes = list(target.size())
        num = np.prod(sizes[1:])
        loss = self.loss(input, target).view(sizes[0], -1)
        u, v = torch.topk(loss, int(self.top_k * num))
        if self.reduce:
            return torch.mean(u)
        else:
            return torch.mean(u, dim=1)


class TopkBCELogits(torch.nn.Module):
    def __init__(self, weight=None, top_k=0.5, reduce=True):
        super(TopkBCELogits, self).__init__()
        self.reduce = reduce
        if weight is None:
            self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='none')
        self.top_k = top_k

    def forward(self, input, target):
        sizes = list(target.size())
        num = np.prod(sizes[1:])
        loss = self.loss(input, target).view(sizes[0], -1)
        u, v = torch.topk(loss, int(self.top_k * num))
        if self.reduce:
            return torch.mean(u)
        else:
            return torch.mean(u, dim=1)
