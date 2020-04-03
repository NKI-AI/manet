# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import os
import numpy as np
import logging
import time
import torch
import apex
import pathlib
import random

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from apex import amp

from config.base_config import cfg, cfg_from_file
from config.base_args import Args
from manet.nn.common.tensor_ops import reduce_tensor_dict
from manet.nn.training.lr_scheduler import WarmupMultiStepLR, build_optim
from manet.nn.common.losses import HardDice
from manet.nn.common.model_utils import load_model, save_model
from manet.data.mammo_data import MammoDataset
from manet.data.transforms import CropAroundBbox, RandomLUT, RandomShiftBbox, RandomFlipTransform
from fexp.transforms import Compose, ClipAndScale
from manet.nn.unet.unet_fastmri_facebook import UnetModel2d
from manet.nn.unet.unet_classifier import UnetModel2dClassifier
from manet.nn.training.sampler import build_sampler
from manet.sys.logging import setup
from manet.sys import multi_gpu
from manet.utils import ensure_list
from fexp.plotting import plot_2d

import torch.nn.functional as F

from fexp.utils.io import read_list, read_json

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

np.random.seed(3145)
random.seed(3145)
torch.manual_seed(3145)


def train_epoch(args, epoch, model, data_loader, optimizer, lr_scheduler, writer, use_classifier=False):
    segmentation_path = pathlib.Path(args.experiment_directory) / args.name / 'segmentations'
    
    model.train()
    avg_loss = 0.
    avg_dice = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    loss_fn = build_losses(use_classifier)
    dice_fn = HardDice(cls=1, binary_cls=True)
    optimizer.zero_grad()

    for iter_idx, batch in enumerate(data_loader):
        images = batch['mammogram'].to(args.device)
        masks = batch['mask'].to(args.device)

        ground_truth = [masks]
        if use_classifier:
            ground_truth += batch['class'].to(args.device)

        # Log first batch to tensorboard
        if iter_idx == 0 and epoch == 0:
            logger.info(f'Logging first batch to Tensorboard.')
            logger.info(f"Image filenames: {batch['image_fn']}")
            logger.info(f"Mask filenames: {batch['label_fn']}")

            image_arr = images.detach().cpu().numpy()[0, 0, ...]
            masks_arr = masks.detach().cpu().numpy()[0, ...]
            logger.info(f'Image min: {image_arr.min()} Image max: {image_arr.max()}')

            # image_grid = torchvision.utils.make_grid(images)
            # mask_grid = torchvision.utils.make_grid(masks)

            # writer.add_image('images', image_grid, 0)
            # writer.add_image('masks', mask_grid, 0)
            # writer.add_graph(model, images.detach().cpu())

            first_image = plot_2d(image_arr, mask=masks_arr)
            first_image.save(segmentation_path / 'first_image.png')
                        
            plot_overlay = torch.from_numpy(np.array(first_image))
            #plot_overlay = torch.from_numpy(np.array(plot_2d(image_arr, mask=masks_arr)))
            writer.add_image('train/overlay', plot_overlay, epoch, dataformats='HWC')

        train_loss = torch.tensor(0.).to(args.device)

        output = ensure_list(model(images))

        #losses = torch.tensor([loss_fn[idx](output[idx], ground_truth[idx]) for idx in range(len(output))]).to(args.device)
        losses = [loss_fn[idx](output[idx], ground_truth[idx]) for idx in range(len(output))]
        #losses.requires_grad_(True)
        train_loss += sum(losses)
        #train_loss.requires_grad_(True)

        # Backprop the loss, use APEX if necessary
        if cfg.APEX >= 0:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            train_loss.backward()
        if torch.isnan(train_loss).any():
            logger.critical(f'Nan loss detected. Stopping training.')
            sys.exit()
        mem_usage = int(torch.cuda.memory_allocated())

        # Gradient accumulation
        if (iter_idx + 1) % cfg.GRAD_STEPS == 0:
            param_lst = amp.master_params(optimizer) if cfg.APEX >= 0 else model.parameters()
            if cfg.GRAD_STEPS > 1:
                for param in param_lst:
                    if param.grad is not None:
                        param.grad.div_(cfg.GRAD_STEPS)
            if cfg.GRAD_CLIP > 0.0:
                torch.nn.utils.clip_grad_norm_(param_lst, cfg.GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_dice = dice_fn(F.softmax(output[0], 1)[0, 1, ...], masks)
        avg_loss = (iter_idx * avg_loss + train_loss.item()) / (iter_idx + 1) if iter_idx > 0 else train_loss.item()
        avg_dice = (iter_idx * avg_dice + train_dice.item()) / (iter_idx + 1) if iter_idx > 0 else train_dice.item()
        metric_dict = {'TrainLoss': train_loss, 'TrainDice': train_dice}

        if cfg.MULTIGPU == 2:
            reduce_tensor_dict(metric_dict)
        if args.local_rank == 0:
            for key in metric_dict:
                writer.add_scalar(key, metric_dict[key].item(), global_step + iter_idx)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step + iter_idx)

        loss_str = f'Loss = {train_loss.item():.4f} Avg Loss = {avg_loss:.4f} '
        for loss_idx, loss in enumerate(losses):
            loss_str += f'Loss_{loss_idx} = {loss.item():.4f} '

        if iter_idx % cfg.REPORT_INTERVAL == 0:
            logger.info(
                f'Ep = [{epoch + 1:3d}/{cfg.N_EPOCHS:3d}] '
                f'It = [{iter_idx + 1:4d}/{len(data_loader):4d}] '
                f'{loss_str}'
                f'Dice = {train_dice.item():.3f} Avg DICE = {avg_dice:.3f} '
                f'Mem = {mem_usage / (1024 ** 3):.2f}GB '
                f'GPU {args.local_rank} '
                f'LR = {optimizer.param_groups[0]["lr"], global_step + iter_idx}'
            )

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, exp_path, return_losses=False, use_classifier=False):
    segmentation_path = pathlib.Path(args.experiment_directory) / args.name / 'segmentations'
    logger.info(f'Evaluation for epoch {epoch}')
    model.eval()

    losses = []
    dices = []
    start = time.perf_counter()
    loss_fn = build_losses(use_classifier)
    dice_fn = HardDice(cls=1, binary_cls=True)

    with torch.no_grad():
        for iter_idx, batch in enumerate(data_loader):
            images = batch['mammogram'].to(args.device)
            masks = batch['mask'].to(args.device)

            ground_truth = [masks]
            if use_classifier:
                ground_truth += batch['class'].to(args.device)

            output = ensure_list(model(images))

            output_softmax = [F.softmax(output[idx], 1) for idx in range(len(output))][0]

            if iter_idx < 1:
                # TODO: Multiple images, using a gridding function.
                image_arr = images.detach().cpu().numpy()[0, 0, ...]
                output_arr = output_softmax.detach().cpu().numpy()[0, 1, ...]
                masks_arr = masks.detach().cpu()[0, ...]

                plot_image = torch.from_numpy(np.array(plot_2d(image_arr)))
                plot_gt = torch.from_numpy(np.array(plot_2d(image_arr, mask=masks_arr)))
                plot_heatmap = torch.from_numpy(np.array(plot_2d(output_arr)))
                plot_overlay = torch.from_numpy(
                    np.array(plot_2d(
                        image_arr, mask=output_arr, overlay_threshold=0.25, overlay_alpha=0.5)))

                #plot_overlay.save(segmentation_path / f'image_{epoch}.png')
                overlay_image = plot_2d(image_arr, mask=output_arr)
                overlay_image.save(segmentation_path / 'overlay_image.png')
                        
                writer.add_image('validation/image', plot_image, epoch, dataformats='HWC')
                writer.add_image('validation/ground_truth', plot_gt, epoch, dataformats='HWC')
                writer.add_image('validation/heatmap', plot_heatmap, epoch, dataformats='HWC')
                writer.add_image('validation/overlay', plot_overlay, epoch, dataformats='HWC')

            batch_losses = torch.tensor([loss_fn[idx](output[idx], ground_truth[idx]) for idx in range(len(output))])
            losses.append(batch_losses.sum().item())

            batch_dice = dice_fn(output_softmax[0, 1, ...], masks)
            dices.append(batch_dice.item())
            del output

    metric_dict = {'DevLoss': torch.tensor(np.mean(losses)).to(args.device),
                   'DevDice': torch.tensor(np.mean(dices)).to(args.device)}

    if cfg.MULTIGPU == 2:
        torch.cuda.synchronize()
        reduce_tensor_dict(metric_dict)
    if args.local_rank == 0:
        for key in metric_dict:
            writer.add_scalar(key, metric_dict[key].item(), epoch)

    torch.cuda.empty_cache()
    if return_losses:
        return metric_dict['DevLoss'].item(), metric_dict['DevDice'], time.perf_counter() - start, losses
    else:
        return metric_dict['DevLoss'].item(), metric_dict['DevDice'], time.perf_counter() - start


def build_losses(use_classifier=False):
    loss_fns = [torch.nn.CrossEntropyLoss(weight=None, reduction='mean')]

    if use_classifier:
        loss_fns += torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

    return loss_fns


def build_model(device, use_classifier=False):
    if use_classifier:
        model = None
    else:
        model = UnetModel2d(
            1, 2, (1024, 1024), 64, 4, 0.1).to(device)

    return model


def build_datasets(data_source):
    # Assume the description file, a training set and a validation set are linked in the main directory.
    train_list = read_list(data_source / 'training_set.txt')
    validation_list = read_list(data_source / 'validation_set.txt')

    mammography_description = read_json(data_source / 'dataset_description.json')

    training_description = {k: v for k, v in mammography_description.items() if k in train_list}
    validation_description = {k: v for k, v in mammography_description.items() if k in validation_list}

    return training_description, validation_description


def build_transforms():
    # Build datasets
    train_transforms = Compose([
        RandomLUT(),
        # ClipAndScale(None, None, [0, 1]),
        RandomShiftBbox([100, 100]),
        CropAroundBbox((1, 1024, 1024)),
        RandomFlipTransform(0.5),
    ])

    validation_transforms = Compose([
        RandomLUT(),
        # ClipAndScale(None, None, [0, 1]),
        CropAroundBbox((1, 1024, 1024))
    ])

    return train_transforms, validation_transforms


def build_samplers(training_set, validation_set, use_weights):
    # Build samplers
    # TODO: Build a custom sampler which can be set differently.
    is_distributed = cfg.MULTIGPU == 2
    if use_weights:
        train_sampler = build_sampler(
            # TODO: Weights
            training_set, 'weighted_random', weights=None, is_distributed=is_distributed)
    else:
        train_sampler = build_sampler(training_set, 'random', weights=False, is_distributed=is_distributed)
    validation_sampler = build_sampler(validation_set, 'sequential', weights=False, is_distributed=is_distributed)

    return train_sampler, validation_sampler


def init_train_data(args, cfg, data_source, use_weights=True):
    training_description, validation_description = build_datasets(data_source)
    train_transforms, validation_transforms = build_transforms()

    training_set = MammoDataset(training_description, data_source, transform=train_transforms, cache_dir='/tmp/train')
    validation_set = MammoDataset(validation_description, data_source, transform=validation_transforms, cache_dir='/tmp/validate')
    logger.info(f'Train dataset size: {len(training_set)}. '
                f'Validation data size: {len(validation_set)}.')

    train_sampler, validation_sampler = build_samplers(training_set, validation_set, use_weights)

    train_loader = DataLoader(
        dataset=training_set,
        batch_size=cfg.BATCH_SZ,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.BATCH_SZ,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, train_sampler, training_set, eval_loader, validation_sampler


def update_train_sampler(args, epoch, model, cfg, dataset, writer, exp_path):
    is_distr = (cfg.MULTIGPU == 2)
    eval_sampler = build_sampler(dataset, 'sequential', weights=False, is_distributed=is_distr)
    eval_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.BATCH_SZ,
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loss, dev_dice, dev_time, losses = evaluate(
        args, epoch, model, eval_loader, writer, exp_path, return_losses=True)

    idx_losses = list(enumerate(losses))
    idx_losses = sorted(idx_losses, key=lambda v: -v[1])
    new_weights = np.ones(len(dataset))
    PERCENTILE = 0.1
    SCALE = 2.0

    for idx in range(int(len(idx_losses) * PERCENTILE)):
        n_idx, _ = idx_losses[idx]
        new_weights[n_idx] *= SCALE
        new_weights /= len(new_weights)

    train_sampler = build_sampler(dataset, 'weighted_random', weights=new_weights, is_distributed=is_distr)
    if cfg.MULTIGPU == 2:
        train_sampler.set_epoch(epoch)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.BATCH_SZ,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, train_sampler


def main(args):
    args.name = args.name if args.name is not None else os.path.basename(args.cfg)[:-5]
    print(f'Run name {args.name}')
    print(f'Local rank {args.local_rank}')
    print(f'Loading config file {args.cfg}')
    cfg_from_file(args.cfg)
    exp_path = args.experiment_directory / args.name
    if args.local_rank == 0:
        print('Creating directories.')
        os.makedirs(cfg.INPUT_DIR, exist_ok=True)
        os.makedirs(exp_path, exist_ok=True)
        os.makedirs(os.path.join(exp_path, 'segmentations'), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(exp_path, 'summary'))
    else:
        time.sleep(1)
        writer = None
    log_name = args.name + f'_{args.local_rank}.log'
    print('Logging into {}'.format(os.path.join(exp_path, log_name)))
    setup(filename=os.path.join(exp_path, log_name), redirect_stderr=False, redirect_stdout=False,
          log_level=logging.INFO if not args.debug else logging.DEBUG)
    logger.info(vars(args))

    if cfg.MULTIGPU == 2:
        logger.info('Initializing process groups.')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        logger.info(f'Synchronizing GPU {args.local_rank}.')
        multi_gpu.synchronize()

    logger.info('Building model.')
    model = build_model(args.device)
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.debug(model)
    logger.info(f'Number of parameters: {n_params} ({n_params / 10.0 ** 3:.2f}k)')
    logger.info('Building optimizers.')
    optimizer = build_optim(model.parameters(), cfg)
    torch.cuda.empty_cache()

    # Initialize APEX if necessary
    if cfg.APEX >= 0:
        opt_level = f'O{cfg.APEX}'
        logger.info(f'Using apex level {opt_level}')
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # Create dataset and initializer LR scheduler
    logger.info('Creating datasets.')
    train_loader, train_sampler, train_set, eval_loader, eval_sampler = init_train_data(args, cfg, args.data_source, use_weights=False)
    solver_steps = [_ * len(train_loader) for _ in
                    range(cfg.LR_STEP_SIZE, cfg.N_EPOCHS, cfg.LR_STEP_SIZE)]
    lr_scheduler = WarmupMultiStepLR(optimizer, solver_steps, cfg.LR_GAMMA, warmup_factor=1 / 10.,
                                     warmup_iters=int(0.5 * len(train_loader)), warmup_method='linear')
    # Load model
    start_epoch = load_model(args, exp_path, model, optimizer, lr_scheduler)
    epoch = start_epoch
    logger.info(f'Starting at epoch {epoch}.')

    # Parallelize model
    if cfg.MULTIGPU == 2:
        if cfg.APEX >= 0:
            logger.info('Using APEX Distributed Data Parallel')
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=cfg.DELAY_REDUCE)
        else:
            logger.info('Using Torch Distributed Data Parallel')
            model = torch.nn.parallel.distributed.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                                          output_device=args.local_rank)
    elif cfg.MULTIGPU == 1:
        logger.info('Using Torch Data Parallel')
        model = torch.nn.DataParallel(model)

    # Train model if necessary
    if args.train:
        for epoch in range(start_epoch, cfg.N_EPOCHS):
            if cfg.MULTIGPU == 2:
                train_sampler.set_epoch(epoch)

            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, lr_scheduler, writer)

            dev_loss, dev_dice, dev_time = evaluate(
                args, epoch, model, eval_loader, writer, exp_path, return_losses=False)

            if args.local_rank == 0:
                save_model(args, exp_path, epoch, model, optimizer, lr_scheduler)
                logger.info(
                    f'Epoch = [{epoch + 1:4d}/{cfg.N_EPOCHS:4d}] TrainLoss = {train_loss:.4g} '
                    f'DevLoss = {dev_loss:.4g} DevDice = {dev_dice:.4g} '
                    f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
                )
            # if (cfg.HARD_MINING_FREQ > 0) and (epoch + 1) % cfg.HARD_MINING_FREQ == 0:
            #     del train_loader, train_sampler
            #     logger.info('Updating samplers for hard mining.')
            #     train_loader, train_sampler = update_train_sampler(args, epoch, model, cfg, train_set, writer, exp_path)

    # Test model if necessary
    if args.test:
        dev_loss, dev_dice, dev_time = evaluate(args, epoch, model, eval_loader, writer, exp_path)
        logger.info(
            f'Epoch = [{epoch:4d}/{cfg.N_EPOCHS:4d}] '
            f'DevLoss = {dev_loss:.4g} DevDice = {dev_dice:.4g} '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


if __name__ == '__main__':
    args = Args().parse_args()
    main(args)
