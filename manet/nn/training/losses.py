"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch

def build_losses(use_classifier=False, multipliers=[1.0, 0.5], top_k=[0.05, None]):
    reduction = ['mean' if _top_k is None else 'none' for _top_k in top_k]

    loss_fns = [torch.nn.CrossEntropyLoss(weight=None, reduction=reduction[0])]

    if use_classifier:
        loss_fns += [torch.nn.CrossEntropyLoss(weight=None, reduction=reduction[1])]
        multipliers = multipliers[1]
    else:
        multipliers = multipliers[0]

    print(multipliers)

    # if args.topk > 0.0 or args.randomk > 0.0:
    #     tensor_size = list(train_loss.size())
    #     num = np.prod(tensor_size[1:])
    #
    # if args.topk > 0.0:
    #     else:
    #     train_loss = train_loss.view(tensor_size[0], -1)
    #     u, uidx = torch.topk(train_loss, int(args.topk * num))
    #     train_loss = torch.mean(u)

    return loss_fns, multipliers


