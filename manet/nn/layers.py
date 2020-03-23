"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch


class GradMultiplication(torch.autograd.Function):
    """
    In the forward pass acts as an identity, but in the backward pass, scales the gradient by a constant factor.
    This can be e.g., useful when attaching a classifier in the bottleneck of a u-net, when one wants to
    not propagate too much of the error signal back into the encoder part.
    """
    def __init__(self, gamma):
        """

        Parameters
        ----------
        gamma : float
            Gradient multiplication factor
        """
        self.gamma = gamma

    # TODO: Does not match signature of Function
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return self.gamma * grad_output
