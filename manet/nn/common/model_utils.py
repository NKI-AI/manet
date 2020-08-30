"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.__init__ import OrderedDict
import torch
import logging
logger = logging.getLogger(__name__)


def count_parameters(params):
    return sum(p.numel() for p in params)


def _remove_module_from_ordered_dict(ordered_dict):
    new_ordered_dict = OrderedDict()
    for idx, (k, v) in enumerate(ordered_dict.items()):
        if k.startswith('module.'):
            if idx == 0:
                logger.debug('Weights start with `.module`, '
                             'suggesting model was saved with DataParallel. Removing these.')

            name = k[7:]
            new_ordered_dict[name] = v
        else:
            new_ordered_dict[k] = v
    return new_ordered_dict


def save_model(exp_dir, epoch, model, optimizer, lr_scheduler, name='model'):
    model_state_dict = model.state_dict()
    model_state_dict = _remove_module_from_ordered_dict(model_state_dict)

    save_dict = {
        'epoch': epoch,
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'exp_dir': exp_dir
    }

    torch.save(save_dict, f=exp_dir / f'{name}_{epoch + 1}.pt')

    with open(exp_dir / 'last_model.txt', 'w') as f:
        f.write(str(epoch + 1))


def load_model(exp_dir, model, optimizer, lr_scheduler, resume, name='model', checkpoint_fn=None):
    start_epoch = 0
    last_model_text_path = exp_dir / 'last_model.txt'
    if resume:
        logger.info('Trying to resume training...')
        if last_model_text_path.exists():
            with open(last_model_text_path, 'r') as f:
                last_epoch = int(f.readline())
            checkpoint_fn = exp_dir / f'{name}_{last_epoch}.pt'
            logger.info(f'Resuming from {checkpoint_fn}.')
        else:
            logger.info('Model not found!')

    if checkpoint_fn:
        logger.info(f'Loading model {checkpoint_fn}.')
        checkpoint = torch.load(checkpoint_fn, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        if resume and last_model_text_path.exists():
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    return start_epoch
