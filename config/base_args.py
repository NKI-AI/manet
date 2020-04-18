# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument('data_source', type=pathlib.Path, help='Path to the dataset')
        self.add_argument('experiment_directory',  type=pathlib.Path, help='Path to the experiment directory')
        self.add_argument('--debug', action='store_true', help='If set debug output will be shown')
        self.add_argument('--local_rank', dest='local_rank', help='Which is the local GPU. Set by init script.',
                            type=int, default=0)
        self.add_argument('--device', type=str, default='cuda',
                            help='Which device to train on. Set to "cuda" to use the GPU')
        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--num-workers', type=int, default=4, help='Number of workers')
        self.add_argument('--cfg', help='Config file for training (and optionally testing)', required=True, type=str)
        self.add_argument('--checkpoint', type=pathlib.Path,
                          help='Path to an existing checkpoint. Used optionally along with "--resume"')
        self.add_argument('--name', help='Run name, if None use config name.', default=None, type=str)

        self.add_argument('--fold', type=int, help='Fold number, will read training and validation set from /fold_idx')
        self.add_argument('--no-rsync', dest='no_rsync', help='use symbolic links instead', action='store_true')
        self.add_argument('--baseline', help='load baseline', action='store_true')
        self.add_argument('--train', help='train model', action='store_true')
        self.add_argument('--resume', help='resume training', action='store_true')
        self.add_argument('--summary', help='make summaries', action='store_true')
        self.add_argument('--test', help='test model', action='store_true')

        self.set_defaults(**overrides)
