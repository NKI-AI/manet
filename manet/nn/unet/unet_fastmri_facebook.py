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


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_channels, out_channels, dropout_prob):
        """

        Parameters
        ----------
        in_channels : int
            Number of channels in the input.
        out_channels : int
            Number of channels in the output.
        dropout_prob : float
            Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_ch = out_channels
        self.dropout_prob = dropout_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, input):
        """

        Parameters
        ----------
        input: torch.Tensor
            Input tensor of shape [batch_size, self.num_channels, height, width]

        Returns
        -------
        torch.Tensor: Output tensor of shape [batch_size, self.out_ch, height, width]

        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(num_channels={self.in_channels}, out_ch={self.out_ch}, ' \
            f'drop_prob={self.dropout_prob})'


class UnetModel2d(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, num_channels, num_classes, output_shape, num_filters, depth, dropout_prob):
        """

        Parameters
        ----------

        num_channels: int
            Number of channels in the input to the U-Net model.
        num_classes: int
            Number of channels in the output to the U-Net model.
        output_shape : list
            Required shape of the output
        num_filters: int
            Number of output channels of the first convolution layer.
        depth: int
            Number of down-sampling and up-sampling layers.
        dropout_prob: float
            Dropout probability.
        """
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.num_filters = num_filters
        self.depth = depth
        self.dropout_prob = dropout_prob

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
            nn.Conv2d(ch // 2, num_classes, kernel_size=1),
            nn.Conv2d(num_classes, num_classes, kernel_size=1),
        )

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

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

    @property
    def shape_in(self):
        return self.output_shape
