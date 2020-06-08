"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def build_losses(use_classifier=False, multipliers=(1.0, 0.5), loss_name='basic', **kwargs):
    multipliers = [torch.tensor(_) for _ in multipliers]

    if loss_name == 'basic':
        loss_fns = [lambda x, y: multipliers[0].to(x.device) * torch.nn.CrossEntropyLoss(weight=None, reduction='mean')(x, y)]

        if use_classifier:
            loss_fns.append(lambda x, y: multipliers[1].to(x.device) * torch.nn.CrossEntropyLoss(weight=None, reduction='mean')(x, y))

    elif loss_name == 'topk':
        loss_fns = [lambda x, y: multipliers[0].to(x.device) * CrossEntropyTopK(top_k=kwargs.get('top_k', 1.0))(x, y)]

        if use_classifier:
            loss_fns.append(
                lambda x, y: multipliers[1].to(x.device) * FocalLoss(gamma=kwargs.get('gamma', 1.0), reduction='mean')(x, y))

    return loss_fns


class CrossEntropyTopK(nn.Module):
    def __init__(self, weight=None, top_k=0.5):
        super(CrossEntropyTopK, self).__init__()
        if not weight:
            self.loss = nn.CrossEntropyLoss(reduction='none')
        else:
            normalized_weight = weight / torch.mean(weight)
            self.loss = torch.nn.CrossEntropyLoss(weight=normalized_weight, reduction='none')
        self.top_k = top_k

    def forward(self, input, target):
        sizes = list(target.size())
        num = np.prod(sizes[1:])
        loss = self.loss(input, target).view(sizes[0], -1)
        u, v = torch.topk(loss, int(self.top_k * num))
        out = torch.mean(u)
        return out


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )