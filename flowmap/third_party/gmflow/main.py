import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import flow_loss_func
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


def main(args):
    if not args.eval and not args.submission and args.inference_dir is None:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            print(args)
            misc.save_args(args)
            misc.check_path(args.checkpoint_dir)
            misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    if not args.eval and not args.submission and not args.inference_dir:
        print('Model definition:')
        print(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # evaluate
    if args.eval:
        val_results = {}

        if 'chairs' in args.val_dataset:
            results_dict = validate_chairs(model_without_ddp,
                                           with_speed_metric=args.with_speed_metric,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )

            val_results.update(results_dict)

        if 'things' in args.val_dataset:
            results_dict = validate_things(model_without_ddp,
                                           padding_factor=args.padding_factor,
                                           with_speed_metric=args.with_speed_metric,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )
            val_results.update(results_dict)

        if 'sintel' in args.val_dataset:
            results_dict = validate_sintel(model_without_ddp,
                                           count_time=args.count_time,
                                           padding_factor=args.padding_factor,
                                           with_speed_metric=args.with_speed_metric,
                                           evaluate_matched_unmatched=args.evaluate_matched_unmatched,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )
            val_results.update(results_dict)

        if 'kitti' in args.val_dataset:
            results_dict = validate_kitti(model_without_ddp,
                                          padding_factor=args.padding_factor,
                                          with_speed_metric=args.with_speed_metric,
                                          attn_splits_list=args.attn_splits_list,
                                          corr_radius_list=args.corr_radius_list,
                                          prop_radius_list=args.prop_radius_list,
                                          )
            val_results.update(results_dict)

        if args.save_eval_to_file:
            misc.check_path(args.checkpoint_dir)
            val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
            with open(val_file, 'a') as f:
                f.write('\neval results after training done\n\n')
                metrics = ['chairs_epe', 'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                           'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40', 'things_clean_s40+',
                           'things_final_epe', 'things_final_s0_10', 'things_final_s10_40', 'things_final_s40+',
                           'sintel_clean_epe', 'sintel_clean_s0_10', 'sintel_clean_s10_40', 'sintel_clean_s40+',
                           'sintel_final_epe', 'sintel_final_s0_10', 'sintel_final_s10_40', 'sintel_final_s40+',
                           'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                           ]
                eval_metrics = []
                for metric in metrics:
                    if metric in val_results.keys():
                        eval_metrics.append(metric)

                metrics_values = [val_results[metric] for metric in eval_metrics]

                num_metrics = len(eval_metrics)

                # save as markdown format
                f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                f.write('\n\n')

        return

    # Sintel and KITTI submission
    if args.submission:
        # NOTE: args.val_dataset is a list
        if args.val_dataset[0] == 'sintel':
            create_sintel_submission(model_without_ddp,
                                     output_path=args.output_path,
                                     padding_factor=args.padding_factor,
                                     save_vis_flow=args.save_vis_flow,
                                     no_save_flo=args.no_save_flo,
                                     attn_splits_list=args.attn_splits_list,
                                     corr_radius_list=args.corr_radius_list,
                                     prop_radius_list=args.prop_radius_list,
                                     )
        elif args.val_dataset[0] == 'kitti':
            create_kitti_submission(model_without_ddp,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    save_vis_flow=args.save_vis_flow,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    )
        else:
            raise ValueError(f'Not supported dataset for submission')

        return

    # inferece on a dir
    if args.inference_dir is not None:
        inference_on_dir(model_without_ddp,
                         inference_dir=args.inference_dir,
                         output_path=args.output_path,
                         padding_factor=args.padding_factor,
                         inference_size=args.inference_size,
                         paired_data=args.dir_paired_data,
                         save_flo_flow=args.save_flo_flow,
                         attn_splits_list=args.attn_splits_list,
                         corr_radius_list=args.corr_radius_list,
                         prop_radius_list=args.prop_radius_list,
                         pred_bidir_flow=args.pred_bidir_flow,
                         fwd_bwd_consistency_check=args.fwd_bwd_consistency_check,
                         )

        return

    # training datset
    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    # Multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)

    total_steps = start_step
    epoch = start_epoch
    print('Start training')

    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]

            results_dict = model(img1, img2,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 )

            flow_preds = results_dict['flow_preds']

            loss, metrics = flow_loss_func(flow_preds, flow_gt, valid,
                                           gamma=args.gamma,
                                           max_flow=args.max_flow,
                                           )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics)

                logger.add_image_summary(img1, img2, flow_preds, flow_gt)

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps % args.val_freq == 0:
                print('Start validation')

                val_results = {}
                # support validation on multiple datasets
                if 'chairs' in args.val_dataset:
                    results_dict = validate_chairs(model_without_ddp,
                                                   with_speed_metric=args.with_speed_metric,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'things' in args.val_dataset:
                    results_dict = validate_things(model_without_ddp,
                                                   padding_factor=args.padding_factor,
                                                   with_speed_metric=args.with_speed_metric,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'sintel' in args.val_dataset:
                    results_dict = validate_sintel(model_without_ddp,
                                                   count_time=args.count_time,
                                                   padding_factor=args.padding_factor,
                                                   with_speed_metric=args.with_speed_metric,
                                                   evaluate_matched_unmatched=args.evaluate_matched_unmatched,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'kitti' in args.val_dataset:
                    results_dict = validate_kitti(model_without_ddp,
                                                  padding_factor=args.padding_factor,
                                                  with_speed_metric=args.with_speed_metric,
                                                  attn_splits_list=args.attn_splits_list,
                                                  corr_radius_list=args.corr_radius_list,
                                                  prop_radius_list=args.prop_radius_list,
                                                  )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if args.local_rank == 0:
                    logger.write_dict(val_results)

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)
                        if args.evaluate_matched_unmatched:
                            metrics = ['chairs_epe',
                                       'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                                       'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40',
                                       'things_clean_s40+',
                                       'sintel_clean_epe', 'sintel_clean_matched', 'sintel_clean_unmatched',
                                       'sintel_clean_s0_10', 'sintel_clean_s10_40',
                                       'sintel_clean_s40+',
                                       'sintel_final_epe', 'sintel_final_matched', 'sintel_final_unmatched',
                                       'sintel_final_s0_10', 'sintel_final_s10_40',
                                       'sintel_final_s40+',
                                       'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                                       ]
                        else:
                            metrics = ['chairs_epe', 'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                                       'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40',
                                       'things_clean_s40+',
                                       'sintel_clean_epe', 'sintel_clean_s0_10', 'sintel_clean_s10_40',
                                       'sintel_clean_s40+',
                                       'sintel_final_epe', 'sintel_final_s0_10', 'sintel_final_s10_40',
                                       'sintel_final_s40+',
                                       'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                                       ]

                        eval_metrics = []
                        for metric in metrics:
                            if metric in val_results.keys():
                                eval_metrics.append(metric)

                        metrics_values = [val_results[metric] for metric in eval_metrics]

                        num_metrics = len(eval_metrics)

                        # save as markdown format
                        if args.evaluate_matched_unmatched:
                            f.write(("| {:>25} " * num_metrics + '\n').format(*eval_metrics))
                            f.write(("| {:25.3f} " * num_metrics).format(*metrics_values))
                        else:
                            f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                            f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                        f.write('\n\n')

                model.train()

            if total_steps >= args.num_steps:
                print('Training done')

                return

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
