# encoding: utf-8
"""
Copyright (c) Nikita Moriakov and Jonas Teuwen

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.base_config import cfg, cfg_from_file
from config.base_args import Args
import matplotlib
if (not cfg.DRAW_PLOT) and cfg.SAVE_PLOT:
    matplotlib.use('Agg')
import numpy as np
import logging
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from apex import amp
import apex
import SimpleITK as sitk
#from apex.parallel import DistributedDataParallel
from manet.nn.common.tensor_ops import reduce_tensor_dict
from manet.nn.training.optim import WarmupMultiStepLR, build_optim
from manet.nn.common.losses import TopkCrossEntropy, HardDice, TopkBCELogits
from manet.nn.common.model_utils import load_model, save_model
from manet.data.mammo_data import MammoDataset
from manet.nn.unet.unet2d import UNet
from manet.nn.training.sampler import build_sampler
from manet.sys.logging import setup
from manet.sys.io import link_data, read_list, read_json
from manet.sys import multi_gpu

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def train_epoch(args, epoch, model, data_loader, optimizer, lr_scheduler, writer):
    model.train()
    avg_loss = 0.
    avg_dice = 0.
    start_epoch = time.perf_counter()
    global_step = epoch * len(data_loader)
    loss_fn = {'topkce': TopkCrossEntropy(top_k=cfg.TOPK), 'topkbce': TopkBCELogits(top_k=cfg.TOPK)}[cfg.LOSS]
    dice_fn = HardDice(cls=1, binary_cls=True)
    optimizer.zero_grad()

    for iter_idx, batch in enumerate(data_loader):
        image = batch['image'].to(args.device)
        mask = batch['mask'].to(args.device)
        train_loss = torch.tensor(0.).to(args.device)
        output = torch.squeeze(model(image), dim=1)
        train_loss += loss_fn(output, mask)

        # Backprop the loss, use APEX if necessary
        if cfg.APEX >= 0:
            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            train_loss.backward()
        if torch.isnan(train_loss).any():
            logger.critical(f'Nan loss detected. Stopping training.')
            sys.exit()
        mem_usage = torch.cuda.memory_allocated()

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

        train_dice = dice_fn(output, mask)
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
            logger.info(
                f'Ep = [{epoch:3d}/{cfg.N_EPOCHS:3d}] '
                f'It = [{iter_idx:4d}/{len(data_loader):4d}] '
                f'Loss = {train_loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Dice = {train_dice.item():.4g} Avg DICE = {avg_dice:.4g} '
                f'Mem = {mem_usage / (1024 ** 3):.2f}GB '
                f'GPU{args.local_rank}'
            )

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, write_volumes=True, return_losses=False):
    model.eval()

    losses = []
    dices = []
    start = time.perf_counter()
    loss_fn = {'topkce': TopkCrossEntropy(top_k=cfg.TOPK, reduce=False),
               'topkbce': TopkBCELogits(top_k=cfg.TOPK, reduce=False)}[cfg.LOSS]
    dice_fn = HardDice(cls=1, binary_cls=True, reduce=False)

    out_volumes = {}
    out_meta = {}
    with torch.no_grad():
        for iter_idx, batch in enumerate(data_loader):
            image = batch['image'].to(args.device)
            mask = batch['mask'].to(args.device)
            src_bbox = batch['src_bbox'].cpu().numpy()
            src_volume = batch['src_volume']
            slice_idx = batch['slice_idx'].cpu().numpy()
            output = torch.squeeze(model(image), dim=1)

            batch_loss = loss_fn(output, mask)
            batch_dice = dice_fn(output, mask)

            if write_volumes:
                output_np = np.where(output.detach().cpu().numpy() > 0.5, 1.0, 0.0).astype(np.uint8)
                for i in range(output_np.shape[0]):
                    crops = src_bbox[i, ...]
                    out_crop = output_np[i, 0, crops[2]:-crops[3], crops[4]:-crops[5]]

                    if src_volume[i] not in out_volumes:
                        out_volumes[src_volume[i]] = []
                        out_meta[src_volume[i]] = {metakey: batch[metakey][i, ...] for metakey in
                                                   ['src_origin', 'src_direction', 'src_spacing']}
                    out_volumes[src_volume[i]].append((slice_idx[i], out_crop))
            for loss in batch_loss:
                losses.append(loss.item())
            for dice in batch_dice:
                dices.append(dice.item())
            del output

    if write_volumes:
        for vkey in out_volumes:
            sorted_patches = [v[1] for v in sorted(out_volumes[vkey], key=(lambda x: x[0]))]
            out_seg = np.stack(sorted_patches)
            out_sitk = sitk.GetImageFromArray(out_seg)
            out_sitk.SetOrigin(out_meta[vkey]['src_origin'].numpy())
            out_sitk.SetDirection(out_meta[vkey]['src_direction'].numpy())
            out_sitk.SetSpacing(out_meta[vkey]['src_spacing'].numpy())
            sitk.WriteImage(sitk.Cast(out_sitk, sitk.sitkUInt8), os.path.join(cfg.EXP_DIR, args.name,
                                                                              'segmentations', vkey))

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


def build_model(device):
    model = UNet(
        1, 2, valid=cfg.UNET.VALID_MODE, mode=cfg.UNET.MODE,
        depth=cfg.UNET.DEPTH, dropout_depth=cfg.UNET.DROPOUT_DEPTH,
        dropout_prob=cfg.UNET.DROPOUT_PROB, channels_base=cfg.UNET.CHANNELS_BASE,
        domain_classifier=cfg.UNET.USE_CLASSIFIER, forward_domain_cls=cfg.UNET.FEED_CLS,
        bn_conv_order=cfg.UNET.BN_ORDER).to(device)
    return model


def init_train_data(args, cfg, data_source, use_weights=True):
    # Assume the description file, a training set and a validation set are linked in the main directory.
    train_list = read_list(data_source / 'training_set.txt')
    validation_list = read_list(data_source / 'validation_set.txt')

    mammography_description = read_json(data_source / 'dataset_description.json')

    training_description = {k: v for k, v in mammography_description.items() if k in train_list}
    validation_description = {k: v for k, v in mammography_description.items() if k in validation_list}

    # Build datasets
    train_transforms = None

    train_set = MammoDataset(training_description, transform=train_transforms)
    validation_set = MammoDataset(validation_description)
    logger.info(f'Train dataset size: {len(train_set)} Validation data size: {len(validation_set)}')

    # Build samplers
    # TODO: Build a custom sampler which can have this included.
    is_distributed = cfg.MULTIGPU == 2
    if use_weights:
        train_sampler = build_sampler(
            train_set, 'weighted_random', weights=train_set.base_weights, is_distributed=is_distributed)
    else:
        train_sampler = build_sampler(train_set, 'random', weights=False, is_distributed=is_distributed)
    validation_sampler = build_sampler(validation_set, 'sequential', weights=False, is_distributed=is_distributed)

    train_loader = DataLoader(
        dataset=train_set,
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
    return train_loader, train_sampler, train_set, eval_loader, validation_sampler


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
    dev_loss, dev_dice, dev_time, losses = evaluate(args, epoch, model, eval_loader, writer, exp_path,
                                                    write_volumes=False, return_losses=True)
    new_weights = dataset.get_weights(losses)
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
    print('Run name {}'.format(args.name))
    print('Local rank {}'.format(args.local_rank))
    print('Loading config file {}'.format(args.cfg))
    cfg_from_file(args.cfg)
    exp_path = os.path.join(cfg.EXP_DIR, args.name)
    if args.local_rank == 0:
        print('Creating directories.')
        os.makedirs(cfg.INPUT_DIR, exist_ok=True)
        os.makedirs(exp_path, exist_ok=True)
        os.makedirs(os.path.join(exp_path, 'segmentations'), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(exp_path, 'summary'))
    else:
        time.sleep(1)
        writer = None
    log_name = args.name + '_{}.log'.format(args.local_rank)
    print('Logging into {}'.format(os.path.join(exp_path, log_name)))
    setup(filename=os.path.join(exp_path, log_name), redirect_stderr=False, redirect_stdout=False,
          log_level=logging.INFO if not args.debug else logging.DEBUG)
    logger.info(vars(args))

    logger.info('Linking data')
    if args.local_rank == 0:
        if args.no_rsync:
            logger.info(f'Assuming data is in {cfg.DATA_SOURCE}')
        else:
            link_data(cfg.DATA_SOURCE, cfg.INPUT_DIR)
    data_source = cfg.DATA_SOURCE if args.no_rsync else cfg.INPUT_DIR

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
    logger.info('Creating datasets')
    train_loader, train_sampler, train_set, eval_loader, eval_sampler = init_train_data(args, cfg, data_source)
    solver_steps = [_ * len(train_loader) for _ in
                    range(cfg.LR_STEP_SIZE, cfg.N_EPOCHS, cfg.LR_STEP_SIZE)]
    lr_scheduler = WarmupMultiStepLR(optimizer, solver_steps, cfg.LR_GAMMA, warmup_factor=1 / 10.,
                                     warmup_iters=int(0.5 * len(train_loader)), warmup_method='linear')

    # Load model
    start_epoch = load_model(args, exp_path, model, optimizer, lr_scheduler)
    epoch = start_epoch

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
            dev_loss, dev_dice, dev_time = evaluate(args, epoch, model, eval_loader, writer, exp_path)

            if args.local_rank == 0:
                save_model(args, exp_path, epoch, model, optimizer, lr_scheduler)
                logger.info(
                    f'Epoch = [{epoch:4d}/{cfg.N_EPOCHS:4d}] TrainLoss = {train_loss:.4g} '
                    f'DevLoss = {dev_loss:.4g} DevDice = {dev_dice:.4g} '
                    f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
                )
            if (cfg.HARD_MINING_FREQ > 0) and (epoch + 1) % cfg.HARD_MINING_FREQ == 0:
                del train_loader, train_sampler
                logger.info('Updating samplers for hard mining.')
                train_loader, train_sampler = update_train_sampler(args, epoch, model, cfg, train_set, writer, exp_path)

    # Test model if necessary
    if args.test:
        dev_loss, dev_dice, dev_time = evaluate(args, epoch, model, eval_loader, writer, exp_path)
        logger.info(
            f'Epoch = [{epoch:4d}/{cfg.N_EPOCHS:4d}] '
            f'DevLoss = {dev_loss:.4g} DevDice = {dev_dice:.4g} '
            f'TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )


if __name__ == '__main__':
    args = Args().parse_args()
    main(args)
