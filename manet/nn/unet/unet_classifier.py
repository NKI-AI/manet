"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Some of this code is written by Facebook for the FastMRI challenge and is licensed under the MIT license.
# THe code has been heavily edited, but some parts can be recognized.

import torch
from torch import nn
from torch.nn import functional as F
from manet.nn.unet.unet_fastmri_facebook import ConvBlock
from manet.nn.layers import grad_multiplier


class UnetModel2dClassifier(nn.Module):
    """
    PyTorch implementation of a U-Net model with a classifier attached at the bottleneck.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self,
                 num_channels,
                 num_segmentation_classes,
                 num_classifier_classes,
                 output_shape,
                 num_filters,
                 depth,
                 dropout_prob,
                 classifier_grad_scale=0.5):
        """

        Parameters
        ----------

        num_channels: int
            Number of channels in the input to the U-Net model.
        num_segmentation_classes: int
            Number of channels in the output to the U-Net model.
        num_classifier_classes : int
            Number of classes in the output of the classifier.
        output_shape : list
            Required shape of the output
        num_filters: int
            Number of output channels of the first convolution layer.
        depth: int
            Number of down-sampling and up-sampling layers.
        dropout_prob: float
            Dropout probability.
        classifier_grad_scale : float
            Multiplication of gradient in classifier.
        """
        super().__init__()

        self.num_channels = num_channels
        self.num_segmentation_classes = num_segmentation_classes
        self.num_classifier_classes = num_classifier_classes
        self.output_shape = output_shape
        self.num_filters = num_filters
        self.depth = depth
        self.dropout_prob = dropout_prob
        self.classifier_grad_scale = classifier_grad_scale

        self.down_sample_layers = nn.ModuleList([ConvBlock(num_channels, num_filters, dropout_prob)])
        ch = num_filters
        for i in range(depth - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, dropout_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, dropout_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(depth - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, dropout_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, dropout_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, num_segmentation_classes, kernel_size=1),
            nn.Conv2d(num_segmentation_classes, num_segmentation_classes, kernel_size=1),
        )

        self.classifier = Classifier(
            (depth - 2) * 2,
            num_domains=num_classifier_classes,
            dropout_prob=dropout_prob,
            grad_scale=self.classifier_grad_scale)

    def forward(self, data):
        """

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape [batch_size, self.in_channels, height, width]

        Returns
        -------

        torch.Tensor: Output tensor of shape [batch_size, self.out_channels, height, width]
        """
        stack = []
        output = data
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        classifier_output = self.classifier(stack[-1])

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output), classifier_output

    @property
    def shape_in(self):
        return self.output_shape


class Classifier(nn.Module):
    def __init__(self, in_channels, num_domains=1, dropout_prob=0.1, grad_scale=0.5):
        super().__init__()

        self.extra_conv = nn.Conv2d(512, in_channels, kernel_size=3, padding=1)
        self.grad_scale = grad_scale
        self.conv_block = ConvBlock(in_channels, in_channels, dropout_prob=dropout_prob)
        self.out_conv = nn.Conv2d(in_channels, num_domains, 1)

    def forward(self, x):
        x = self.extra_conv(x)
        x = self.conv_block(grad_multiplier(x, self.grad_scale))
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.out_conv(x)[..., 0, 0]

        return x
