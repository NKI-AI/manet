# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import Sampler
import logging
import math

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """
        TODO: This is from Facebook's maskrcnn_benchmark!!

        Sampler that restricts data loading to a subset of the dataset.
            It is especially useful in conjunction with
            :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
            process can pass a DistributedSampler instance as a DataLoader sampler,
            and load a subset of the original dataset that is exclusive to it.
            .. note::
                Dataset is assumed to be of constant size.

        Parameters
        ----------
        dataset: dataset
            Dataset used for sampling
        num_replicas: int
            Nunmber of processes participating in distributed training
        rank: int
            Rank of the processes within num_replicas
        shuffle: bool
            Shuffle data or sequential
        """
        self.logger = logging.getLogger(type(self).__name__)
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.weights = None

    def __iter__(self):
        if self.shuffle or self.weights is not None:
            # Deterministically shuffle based on epoch
            self.logger.debug(f'Shuffling according to {self.epoch}')
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if not self.weights:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = torch.multinomial(self.weights, len(self.dataset), True, generator=g).tolist()
        else:
            self.logger.debug(f'Shuffling disabled, producing sequential indices.')
            indices = torch.arange(len(self.dataset)).tolist()

        if self.shuffle:  # In case shuffling is enabled, extra samples should be added to make output evenly divisible.
            # Make evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_weights(self, weights):
        raise NotImplementedError(f'Likely to be buggy. Check https://github.com/AIIMLab/DCIS-mammography/issues/3')
        self.weights = torch.tensor(weights, dtype=torch.double)


def _build_sampler(dataset, sampler, weights=False, is_distributed=False):
    if sampler == 'random':
        if is_distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = RandomSampler(dataset)
    elif sampler == 'weighted_random':
        assert (weights is not None), 'for a weighted random sampler weights are needed'
        if is_distributed:
            sampler = DistributedWeightedSampler(dataset)
            sampler.set_weights(weights)
        else:
            sampler = WeightedRandomSampler(weights, len(dataset))
    elif sampler == 'sequential':
        if is_distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = SequentialSampler(dataset)
    else:
        raise NotImplementedError(f'Sampler {sampler} not implemented.')

    return sampler


def build_samplers(training_set, validation_set, use_weights, is_distributed):
    # Build samplers
    # TODO: Build a custom sampler which can be set differently.
    # TODO: Merge with the above function.
    if use_weights:
        train_sampler = _build_sampler(
            # TODO: Weights
            training_set, 'weighted_random', weights=None, is_distributed=is_distributed)
    else:
        train_sampler = _build_sampler(training_set, 'random', weights=False, is_distributed=is_distributed)
    validation_sampler = _build_sampler(validation_set, 'sequential', weights=False, is_distributed=is_distributed)

    return train_sampler, validation_sampler