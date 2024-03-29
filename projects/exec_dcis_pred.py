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
# import apex
import random

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from apex import amp


from omegaconf import OmegaConf
from config.base_config import DefaultConfig, UnetConfig, SolverConfig
from config.base_args import Args
from manet.nn import build_model
from manet.nn.common.tensor_ops import reduce_tensor_dict
from manet.nn.training.losses import build_losses
from manet.nn.training.lr_scheduler import WarmupMultiStepLR, build_optim
from manet.nn.common.losses import HardDice
from manet.nn.common.model_utils import load_model, save_model
from manet.data.mammo_data import MammoDataset, build_datasets
from manet.data.transforms import build_transforms
from manet.nn.training.sampler import build_samplers
from manet.sys.logging import setup
from manet.sys import multi_gpu
from manet.utils import ensure_list
from fexp.plotting import plot_2d
from fexp.utils.io import write_list

from operator import mul
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import functools
import torch.nn.functional as F

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

np.random.seed(3145)
random.seed(3145)
torch.manual_seed(3145)

import torchvision.utils


def log_images_to_tensorboard(writer, epoch, images, masks, output_softmax, overlay_threshold=0.4, grid=(2, 2)):
    ground_truth_list = []
    heatmap_list = []
    overlay_list = []
    for idx in range(functools.reduce(mul, grid)):
        image_arr = images[idx].detach().cpu().numpy()[0, 0, ...]
        masks_arr = masks[idx].detach().cpu()[0, ...]

        plot_gt = torch.from_numpy(np.array(plot_2d(image_arr, mask=masks_arr)))
        if output_softmax is not None:
            output_arr = output_softmax[idx][0].detach().cpu().numpy()[0, 1, ...]
            plot_heatmap = torch.from_numpy(np.array(plot_2d(output_arr)))
            plot_overlay = torch.from_numpy(
                np.array(plot_2d(
                    image_arr, mask=output_arr, overlay_threshold=overlay_threshold, overlay_alpha=0.5)))

        ground_truth_list.append(plot_gt)
        heatmap_list.append(plot_heatmap)
        overlay_list.append(plot_overlay)


    writer.add_images('validation/ground_truth', ground_truth_list[0], epoch, dataformats='HWC')
    writer.add_image('validation/heatmap', heatmap_list[0], epoch, dataformats='HWC')
    writer.add_image('validation/overlay', overlay_list[0], epoch, dataformats='HWC')


def train_epoch(cfg, args, epoch, model, data_loader, optimizer, lr_scheduler, writer, use_classifier=False, debug=False):
    model.train()
    avg_loss = 0.
    avg_dice = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    loss_fn = build_losses(
        use_classifier, loss_name=cfg.network.loss_name, top_k=cfg.network.loss_top_k, gamma=cfg.network.loss_gamma)
    dice_fn = HardDice(cls=1, binary_cls=True)
    optimizer.zero_grad()

    for iter_idx, batch in enumerate(data_loader):
        if debug:
            if iter_idx == 1:
                break

        images = batch['image'].to(args.device)
        masks = batch['mask'].to(args.device)
        ground_truth = [masks]
        if use_classifier:
            ground_truth.append(batch['class'].to(args.device))

        # Log first batch to tensorboard
        if iter_idx == 0 and epoch == 0 and debug:
            logger.info(f'Logging first batch to Tensorboard.')
            logger.info(f"Image filenames: {batch['image_fn']}")
            logger.info(f"Mask filenames: {batch['label_fn']}")

            image_arr = images.detach().cpu().numpy()[0, 0, ...]
            masks_arr = masks.detach().cpu().numpy()[0, ...]

            # image_grid = torchvision.utils.make_grid(images)
            # mask_grid = torchvision.utils.make_grid(masks)

            # writer.add_image('images', image_grid, 0)
            # writer.add_image('masks', mask_grid, 0)
            # writer.add_graph(model, images.detach().cpu())

            plot_overlay = torch.from_numpy(np.array(plot_2d(image_arr, mask=masks_arr)))
            writer.add_image('train/overlay', plot_overlay, epoch, dataformats='HWC')

        output = ensure_list(model(images))
        losses = [loss_fn[idx](output[idx], ground_truth[idx]) for idx in range(len(output))]
        train_loss = sum(losses)

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

        # TODO: Dice function does not accept batches
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

        if iter_idx % cfg.REPORT_INTERVAL == 0:
            loss_str = f'Loss = {train_loss.item():.4f} Avg Loss = {avg_loss:.4f} '
            for loss_idx, loss in enumerate(losses):
                loss_str += f'Loss_{loss_idx} = {loss.item():.4f} '
            logger.info(
                f'Ep = [{epoch + 1:3d}/{cfg.num_epochs:3d}] '
                f'It = [{iter_idx + 1:4d}/{len(data_loader):4d}] '
                f'{loss_str}'
                f'Dice = {train_dice.item():.3f} Avg DICE = {avg_dice:.3f} '
                f'Mem = {mem_usage / (1024 ** 3):.2f}GB '
                f'GPU{args.local_rank}'
            )

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(cfg, args, epoch, model, data_loader, writer, exp_path, use_classifier=False):
    model.eval()

    losses = []
    dices = []
    start = time.perf_counter()
    loss_fn = build_losses(
        use_classifier, loss_name=cfg.network.loss_name, top_k=cfg.network.loss_top_k, gamma=cfg.network.loss_gamma)
    dice_fn = HardDice(cls=1, binary_cls=True)

    aggregate_outputs = [False]
    if use_classifier:
        aggregate_outputs.append(True)

    stored_outputs = []
    stored_groundtruths = []
    stored_filenames = []

    with torch.no_grad():
        log_images = []
        log_masks = []
        log_output = []
        for iter_idx, batch in enumerate(data_loader):
            images = batch['image'].to(args.device)
            masks = batch['mask'].to(args.device)
            filename = batch['image_fn']

            ground_truth = [masks]
            if use_classifier:
                ground_truth.append(batch['class'].to(args.device))

            output = ensure_list(model(images))
            output_softmax = [F.softmax(output[idx], 1) for idx in range(len(output))]

            stored_outputs.append(
                [curr_output if aggregate else None for
                 curr_output, aggregate in zip(output_softmax, aggregate_outputs)])
            stored_groundtruths.append(
                [curr_gtr if aggregate else None for
                 curr_gtr, aggregate in zip(ground_truth, aggregate_outputs)])
            stored_filenames.append(filename)

            if writer:
                if iter_idx < 2*2:
                    log_images.append(images)
                    log_masks.append(masks)
                    log_output.append(output_softmax)

                if iter_idx == 2*2:
                    log_images_to_tensorboard(writer, epoch, log_images, log_masks, log_output, overlay_threshold=0.5)
                    del log_images
                    del log_masks
                    del log_output

            batch_losses = torch.tensor([loss_fn[idx](output[idx], ground_truth[idx]) for idx in range(len(output))])
            losses.append(batch_losses.sum().item())

            batch_dice = dice_fn(output_softmax[0][:, 1, ...], masks)
            dices.append(batch_dice.item())
            del output

    metric_dict = {
       'DevLoss': torch.tensor(np.mean(losses)).to(args.device),
       'DevDice': torch.tensor(np.mean(dices)).to(args.device)
    }

    # Compute metrics for stored output:
    output_result = []
    if use_classifier:
        grab_idx = 1

        outputs = torch.stack([_[grab_idx] for _ in stored_outputs]).cpu().numpy()[:, 0, 1].astype(np.float) # ?

        gtrs = torch.stack([_[grab_idx] for _ in stored_groundtruths]).cpu().numpy()[:, 0].astype(np.float)

        output_result = (stored_filenames, outputs, gtrs)

        auc = roc_auc_score(gtrs, outputs)
        # balanced_accuracy = balanced_accuracy_score(gtrs, outputs, sample_weight=None, adjusted=False)
        #f1_score_val = f1_score(gtrs, outputs)
        logger.info(metric_dict)
        logger.info(auc)
        metric_dict['DevAUC'] = torch.tensor(auc).to(args.device)
        # metric_dict['DevBalancedAcc'] = torch.tensor(balanced_accuracy).to(args.device)
        # metric_dict['DevF1Score'] = torch.tensor(f1_score_val).to(args.device)

    if cfg.MULTIGPU == 2:
        torch.cuda.synchronize()
        reduce_tensor_dict(metric_dict)
    if args.local_rank == 0 and writer:
        for key in metric_dict:
            writer.add_scalar(key, metric_dict[key].item(), epoch)

    metric_string = f''
    for k, v in metric_dict.items():
        metric_string += f'{k} = {v:.4g} '

    logger.info(
        f'Epoch = [{epoch + 1:4d}/{cfg.num_epochs:4d}] '
        f'{metric_string} '
        f'DevTime = {time.perf_counter() - start:.4f}s'
    )

    torch.cuda.empty_cache()

    return metric_dict, output_result


def build_dataloader(batch_size, training_set, training_sampler, validation_set=None, validation_sampler=None):
    training_loader = DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        sampler=training_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if validation_set:
        validation_loader = DataLoader(
            dataset=validation_set,
            batch_size=batch_size,  # TODO: fix this once dice is fixed.
            sampler=validation_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        return training_loader, validation_loader
    return training_loader


def main(args):
    args.name = args.name if args.name is not None else os.path.basename(args.cfg)[:-5]
    base_cfg = OmegaConf.structured(DefaultConfig)
    base_cfg = OmegaConf.merge(base_cfg, {'network': UnetConfig(), 'SOLVER': SolverConfig()})

    cfg = OmegaConf.merge(base_cfg, OmegaConf.load(args.cfg))

    print(f'Run name {args.name}')
    print(f'Local rank {args.local_rank}')
    print(f'Loading config file {args.cfg}')

    exp_path = args.experiment_directory / args.name
    if args.fold:
        exp_path = exp_path / f'fold_{args.fold}'

    if args.local_rank == 0:
        print('Creating directories.')
        os.makedirs(exp_path, exist_ok=True)
        os.makedirs(exp_path / 'segmentations', exist_ok=True)
        writer = SummaryWriter(log_dir=exp_path / 'summary')
    else:
        time.sleep(1)
        writer = None
    log_name = f'log_{args.local_rank}.log'
    print(f'Logging into {exp_path / log_name}')
    setup(filename=exp_path / log_name, redirect_stderr=False, redirect_stdout=False,
          log_level=logging.INFO if not args.debug else logging.DEBUG)
    logger.info(vars(args))
    logger.info(cfg.pretty())

    if cfg.MULTIGPU == 2:
        logger.info('Initializing process groups.')
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        logger.info(f'Synchronizing GPU {args.local_rank}.')
        multi_gpu.synchronize()

    logger.info('Building model.')
    model = build_model(
        use_classifier=cfg.network.use_classifier, num_base_filters=cfg.network.num_base_filters,
        depth=cfg.network.depth, output_shape=cfg.patch_size,
        classifier_grad_scale=cfg.network.classifier_gradient_multiplier).to(args.device)
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
    is_distributed = (cfg.MULTIGPU == 2)

    # Create dataset and initializer LR scheduler
    logger.info('Creating datasets.')
    training_description, validation_description = build_datasets(args.data_source, args.fold)
    training_transforms, validation_transforms = build_transforms()

    training_set = MammoDataset(training_description, args.data_source, transform=training_transforms)
    validation_set = MammoDataset(validation_description, args.data_source, transform=validation_transforms)
    logger.info(f'Train dataset size: {len(training_set)}. '
                f'Validation data size: {len(validation_set)}.')
    training_sampler, validation_sampler = build_samplers(
        training_set, validation_set, use_weights=False, is_distributed=is_distributed)
    training_loader, validation_loader = build_dataloader(
        cfg.batch_size, training_set, training_sampler, validation_set, validation_sampler)

    solver_steps = [_ * len(training_loader) for _ in
                    range(cfg.lr_step_size, cfg.num_epochs, cfg.lr_step_size)]
    lr_scheduler = WarmupMultiStepLR(optimizer, solver_steps, cfg.LR_GAMMA, warmup_factor=1 / 10.,
                                     warmup_iters=int(0.5 * len(training_loader)), warmup_method='linear')

    # Load model
    # TODO: Amp requires loading the state dict
    start_epoch = load_model(exp_path, model, optimizer, lr_scheduler, args.resume, checkpoint_fn=args.checkpoint)
    epoch = start_epoch
    logger.info(f'Starting at epoch {epoch}.')

    # Parallelize model
    if cfg.MULTIGPU == 2:
        if cfg.APEX >= 0:
            logger.info('Using APEX Distributed Data Parallel')
            # TODO: Apex unstable, replace with torch DDP
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
        for epoch in range(start_epoch, cfg.num_epochs):
            if cfg.MULTIGPU == 2:
                training_sampler.set_epoch(epoch)

            train_loss, train_time = train_epoch(cfg, args, epoch, model, training_loader, optimizer, lr_scheduler, writer,
                                                 use_classifier=cfg.network.use_classifier)
            logger.info(f'Evaluation for epoch {epoch + 1}')
            validate_metrics = evaluate(
                cfg, args, epoch, model, validation_loader, writer, exp_path, use_classifier=cfg.network.use_classifier)

            if args.local_rank == 0:
                save_model(exp_path, epoch, model, optimizer, lr_scheduler)

            # if (cfg.HARD_MINING_FREQ > 0) and (epoch + 1) % cfg.HARD_MINING_FREQ == 0:
            #     del train_loader, train_sampler
            #     logger.info('Updating samplers for hard mining.')
            #     train_loader, train_sampler = update_train_sampler(args, epoch, model, cfg, train_set, writer, exp_path)

    writer.close()
    # Test model if necessary
    if args.test:
        logger.info('Validating....')
        validate_metrics, output = evaluate(
            cfg, args, epoch, model, validation_loader, None, exp_path, use_classifier=cfg.network.use_classifier)

        filenames, outputs, gtrs = output
        output_csv = ['filename;output_probability;gtr']
        for filename, output_val, gtr in zip(filenames, outputs.tolist(), gtrs.tolist()):
            csv_str = f'{filename[0]};{output_val};{gtr}'
            output_csv.append(csv_str)

        if not args.fold:
            write_list(exp_path / 'validation_results.csv', output_csv)
        else:
            write_list(exp_path / f'validation_results_{args.fold}.csv', output_csv)

if __name__ == '__main__':
    args = Args().parse_args()
    main(args)
