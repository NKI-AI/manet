"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch


class GradMultiplication(torch.autograd.Function):
    gamma = 1.0
    """
    In the forward pass acts as an identity, but in the backward pass, scales the gradient by a constant factor.
    This can be e.g., useful when attaching a classifier in the bottleneck of a u-net, when one wants to
    not propagate too much of the error signal back into the encoder part.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradMultiplication.gamma * grad_output


def grad_multiplier(x, gamma=0.5):
    GradMultiplication.gamma = gamma
    return GradMultiplication.apply(x)

