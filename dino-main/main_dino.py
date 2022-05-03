# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from tensorboardX import SummaryWriter
import logging
from torch.utils.data import DataLoader
from data.data.cub import CustomCub2011, get_cub_datasets
from data.data.stanford_cars import CarsDataset, get_scar_datasets
from data.data.cifar import get_cifar10_datasets, get_cifar100_datasets
from data.data.tinyimagenet import get_tiny_image_net_datasets

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.utils.linear_assignment_ import linear_assignment
import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--dataset', default='CUB', type=str)
    parser.add_argument('--num_train_classes', default=None, type=int)
    parser.add_argument('--num_aux_classes', default=None, type=int)
    parser.add_argument('--num_test_classes', default=None, type=int)
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] ,
            # \    + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--teacher_ssl', action="store_true", default=False)
    parser.add_argument('--student_ssl', action="store_true", default=False)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--ce_warmup_epoch', default=30, type=int)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00003, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--clus_dim', default=2, type=int)

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./checkpoints/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    args.input_size = 224
    train_transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'CUB':
        args.nb_classes = 100
        args.input_size = 224
        if args.num_train_classes is not None:
            datasets = get_cub_datasets(train_transform, test_transform, seed=args.seed, num_train_classes=args.num_train_classes,
                                        num_auxiliary_classes=args.num_aux_classes, num_open_set_classes=args.num_test_classes)
        else:
            datasets = get_cub_datasets(train_transform, test_transform, seed=args.seed)
    elif args.dataset == 'SCAR':
        args.nb_classes = 98
        args.input_size = 224
        datasets = get_scar_datasets(train_transform, test_transform, seed=args.seed)
    elif args.dataset == 'CIFAR10':
        args.nb_classes = 5
        args.input_size = 32
        # args.batch_size = args.batch_size * 8
        datasets = get_cifar10_datasets(train_transform, test_transform, seed=args.seed)
    elif args.dataset == 'CIFAR100':
        args.nb_classes = 80
        args.input_size = 32
        # args.batch_size = args.batch_size * 8
        datasets = get_cifar100_datasets(train_transform, test_transform, seed=args.seed)
    elif args.dataset == 'TIMNT':
        args.nb_classes = 100
        args.input_size = 64
        datasets = get_tiny_image_net_datasets(train_transform, test_transform, seed=args.seed)

    mix_train_loader = DataLoader(datasets['mix_train'], batch_size=args.batch_size,
                                  shuffle=True, sampler=None, num_workers=args.num_workers)
    mix_test_loader = DataLoader(datasets['mix_test'], batch_size=args.batch_size,
                                 shuffle=False, sampler=None, num_workers=args.num_workers)
    labeled_train_loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      shuffle=True, sampler=None, num_workers=args.num_workers)
    labeled_eval_loader = DataLoader(datasets['val'], batch_size=args.batch_size,
                                     shuffle=False, sampler=None, num_workers=args.num_workers)
    aux_train_loader = DataLoader(datasets['aux_train'], batch_size=args.batch_size,
                                 shuffle=False, sampler=None, num_workers=args.num_workers)
    aux_test_loader = DataLoader(datasets['aux_test'], batch_size=args.batch_size,
                                 shuffle=False, sampler=None, num_workers=args.num_workers)
    test_unknown_loader = DataLoader(datasets['test_unknown'], batch_size=args.batch_size,
                                     shuffle=False, sampler=None, num_workers=args.num_workers)
    test_known_loader = DataLoader(datasets['test_known'], batch_size=args.batch_size,
                                   shuffle=False, sampler=None, num_workers=args.num_workers)
    data_loader_train = labeled_train_loader
    data_loader_val = labeled_eval_loader


    # dataset = datasets.ImageFolder(args.data_path, transform=transform)
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     sampler=sampler,
    #     batch_size=args.batch_size_per_gpu,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # print(f"Data loaded: there are {len(datasets['mix_train'])} images.")

    # global means

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ),
         DINOHead(
             embed_dim,
             args.clus_dim,
             use_bn=args.use_bn_in_head,
             norm_last_layer=args.norm_last_layer,
         )
                                     )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    if args.pretrained_weights != '':
        print("Loading pretrained weights!")
        if args.teacher_ssl:
            utils.load_pretrained_weights(teacher, args.pretrained_weights, 'teacher', args.arch, args.patch_size)
        else:
            utils.load_pretrained_weights(teacher, "imagenet_pretrained.pth", 'teacher', args.arch, args.patch_size)
        if args.student_ssl:
            utils.load_pretrained_weights(student, args.pretrained_weights, 'student', args.arch, args.patch_size)
        else:
            utils.load_pretrained_weights(student, "imagenet_pretrained.pth", 'student', args.arch, args.patch_size)

    # freeze layers in the student model except for head and the last layer in the backbone
    # for name, param in student.named_parameters():
    #     if not '11' in name or not 'head' in name:
    #         param.requires_grad = False

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)

    # teacher and student start with the same weights
    # teacher_without_ddp.load_state_dict(student.module.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(labeled_train_loader) + len(aux_train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(labeled_train_loader) + len(aux_train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(labeled_train_loader ))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    temp_name = "_"
    if args.teacher_ssl:
        temp_name += "tss"
    else:
        temp_name += "ts"
    if args.student_ssl:
        temp_name += "_sss"
    else:
        temp_name += "_ss"
    args.output_dir = os.path.join(args.output_dir,args.dataset+temp_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    # Logger
    # args.output_dir = os.path.join(args.output_dir, args.dataset)
    logger = SummaryWriter(args.output_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(args))


    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        # labeled_train_loader .sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        if epoch < args.ce_warmup_epoch:
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                labeled_train_loader , optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args, logger)
        else:
            means = update_means(student, labeled_train_loader, aux_train_loader, args)
            train_stats = train_one_epoch_clus(student, teacher, teacher_without_ddp, dino_loss,
                                          labeled_train_loader, aux_train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                          epoch, fp16_scaler, args, logger, means)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if epoch % 5 == 0:
            _ = evaluate(test_known_loader, student, dino_loss, epoch, logger, "Val")
            _ = evaluate(aux_test_loader, student, dino_loss, epoch, logger, "Aux")
            _ = evaluate(test_unknown_loader, student, dino_loss, epoch, logger, "Unknown")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, logger):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    niters_per_epoch = len(data_loader)
    idx = 0
    for it, (images, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # print(images[0].shape)
        # print(label)
        idx += 1
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        label = label.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images, clus_fea=False)
            d_loss, ce_loss = dino_loss(student_output, teacher_output, label, epoch)
            logger.add_scalar('Train_Loss/dino_loss', d_loss, epoch * niters_per_epoch + idx)
            logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)

            loss = 0.5 * d_loss + ce_loss


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_loss(student, teacher,teacher_without_ddp, dino_loss, data_loader, optimizer,
               lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args, logger,
               use_ce=False, use_clus=False, aux_set=False, metric_logger=None, header=None, means=None, train_loader_len=None):

    niters_per_epoch = len(data_loader)
    idx = 0
    for it, (images, label) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        idx += 1
        # update weight decay and learning rate according to their schedule
        if aux_set:
            it = (len(data_loader) + train_loader_len)* epoch + it
        else:
            it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        label = label.cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            if use_clus:
                student_output, student_output2 = student(images, clus_fea=True)
            else:
                student_output = student(images)

            d_loss, ce_loss = dino_loss(student_output, teacher_output, label, epoch)
            logger.add_scalar('Train_Loss/dino_loss', d_loss, epoch * niters_per_epoch + idx)
            loss = d_loss
            if use_clus:
                # fea = student.module.forward_features(images)
                fea = student_output2
                if aux_set:
                    targets_temp = torch.Tensor.cpu(label).detach().numpy() + args.num_train_classes
                else:
                    targets_temp = torch.Tensor.cpu(label).detach().numpy()
                targets_temp = np.tile(targets_temp.squeeze(),args.local_crops_number + 2).reshape(-1)
                # print("target_temp:")
                # print(targets_temp)

                fea = fea.unsqueeze(1).repeat_interleave(args.num_train_classes + args.num_aux_classes, dim=1)
                # print(fea.shape)
                # print(means.shape)
                dist = torch.sqrt(torch.sum(torch.square(fea - means), dim=-1))
                # mask = torch.full((args.batch_size * (args.local_crops_number + 2), args.num_train_classes + args.num_aux_classes), -1, dtype=int).cuda()
                mask = torch.full((fea.shape[0], args.num_train_classes + args.num_aux_classes), -1, dtype=int).cuda()
                for id, index in enumerate(targets_temp):
                    mask[id, index] = 1
                # print(dist.shape)
                # print(mask.shape)
                clu_loss = torch.mean(torch.sum(torch.mul(dist, mask), dim=1)) / fea.shape[1]
                logger.add_scalar('Train_Loss/clu_loss', clu_loss, epoch * niters_per_epoch + idx)
                loss += clu_loss

            if use_ce:
                logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)
            # logger.add_scalar('Train_Loss/clu_loss', clu_loss, epoch * niters_per_epoch + idx)
                loss += 0.5 * ce_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        if not aux_set:
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    return None

def train_one_epoch_clus(student, teacher, teacher_without_ddp, dino_loss, train_data_loader, aux_data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, logger, means):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    train_loss(student, teacher, teacher_without_ddp, dino_loss, train_data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args, logger, use_ce=False, use_clus=True, aux_set=False, metric_logger=metric_logger,
                    header=header, means=means)

    train_loader_len = len(train_data_loader)
    # train_loss(student, teacher, teacher_without_ddp, dino_loss, aux_data_loader,
    #                 optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
    #                 fp16_scaler, args, logger, use_ce=False, use_clus=True, aux_set=True, metric_logger=metric_logger,
    #                 header=header, means=means, train_loader_len=train_loader_len)
    train_loss(student, teacher, teacher_without_ddp, dino_loss, aux_data_loader,
               optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
               fp16_scaler, args, logger, use_ce=True, use_clus=False, aux_set=True, metric_logger=metric_logger,
               header=header, means=means, train_loader_len=train_loader_len)

    # means = update_means(student,train_data_loader,aux_data_loader,args)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def update_means(student, train_data_loader, aux_data_loader, args):
    fea_list = None
    target_list = None
    with torch.no_grad():
        for it, (images, label) in enumerate(train_data_loader):
            images = [im.cuda(non_blocking=True) for im in images]
            label = label.cuda(non_blocking=True)

            # fea = student.module.forward_features(images)
            _, fea = student(images, clus_fea=True)
            feas = torch.Tensor.cpu(fea).detach().numpy()
            if fea_list is None:
                fea_list = feas
            else:
                fea_list = np.vstack((fea_list, feas))
            targets_temp = torch.Tensor.cpu(label).detach().numpy()
            if target_list is None:
                target_list = targets_temp
            else:
                target_list = np.concatenate((target_list, targets_temp))

        for it, (images, label) in enumerate(aux_data_loader):
            # move images to gpu
            images = [im.cuda(non_blocking=True) for im in images]
            label = label.cuda(non_blocking=True)

            # fea = student.module.forward_features(images)
            _, fea = student(images, clus_fea=True)
            feas = torch.Tensor.cpu(fea).detach().numpy()
            if fea_list is None:
                fea_list = feas
            else:
                fea_list = np.vstack((fea_list, feas))
            targets_temp = torch.Tensor.cpu(label).detach().numpy() + args.num_train_classes
            if target_list is None:
                target_list = targets_temp
            else:
                target_list = np.concatenate((target_list, targets_temp))

    target_list = target_list.squeeze().repeat(args.local_crops_number + 2)
    target_list = np.expand_dims(target_list, axis=1)
    # print(target_list.shape)
    # print(fea_list.shape)
    fea_tar = np.hstack((target_list, fea_list)).squeeze()
    # print(fea_tar.shape)
    means_dict = {}
    for i in np.unique(fea_tar[:, 0]):
        tmp = fea_list[np.where(fea_tar[:, 0] == i)]
        means_dict[i] = np.mean(tmp, axis=0)
    # print(means[i].shape)
    tmp_means = []
    for k in range(args.num_train_classes + args.num_aux_classes):
        tmp_means.append(means_dict[k])
    means = torch.as_tensor(np.array(tmp_means)).cuda()
    print("The means are updated")
    print("There are {} classes".format(means.shape[0]))
    return means


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # print(w.shape)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    # print(ind.shape)
    # print(sum([w[i, j] for i, j in ind]))
    # for i in range(ind)
#     temp = 0
#     for i in ind[0]:
#         for j in ind[1]:
#             temp += w[i,j]
#     return temp * 1.0 / y_pred.size
#     print(w)
#     print(ind)
#     return w, ind
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

@torch.no_grad()
def evaluate(data_loader, model, dino_loss, epoch, logger=None, name="Val", record=True):
    criterion = dino_loss
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    fea_list = None
    target_list = None

    niters_per_epoch = len(data_loader)
    idx = 0
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        idx += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            # fea = model.module.forward_features(images)
            fea = model(images)
            # d_loss, ce_loss = criterion(outputs, output targets)
            # clu_loss = criterion2(fea, targets)
            # ce_loss = criterion1(samples, outputs, targets)
            # clu_loss = criterion2(samples, outputs, targets)
            # loss = criterion(output, target)
        # if record:
        #     logger.add_scalar('{}/ce_loss'.format(name), ce_loss, epoch * niters_per_epoch + idx)
        # logger.add_scalar('{}/clu_loss'.format(name), clu_loss, epoch * niters_per_epoch + idx)

        feas = torch.Tensor.cpu(fea).detach().numpy()
        if fea_list is None:
            fea_list = feas
        else:
            fea_list = np.vstack((fea_list,feas))
        targets_temp = torch.Tensor.cpu(targets).detach().numpy()
        if target_list is None:
            target_list = targets_temp
        else:
            target_list = np.concatenate((target_list, targets_temp))

        # if name == "Val" and record:
        #     acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        #
        #     batch_size = images.shape[0]
        #     metric_logger.update(loss=ce_loss.item())
        #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    kmeans = KMeans(n_clusters=len(set(target_list)), random_state=1).fit(fea_list)
    # print(target_list.shape)
    # print(len(kmeans.labels_))
    # target_list = target_list.
    cls_acc = cluster_acc(target_list, kmeans.labels_)
    # nmi = nmi_score(target_list, kmeans.labels_)
    # ari = ari_score(target_list, kmeans.labels_)
    if record:
        logger.add_scalar('{}/cls_acc'.format(name), cls_acc, epoch)
        # logger.add_scalar('{}/nmi_score'.format(name), nmi, epoch)
        # logger.add_scalar('{}/ari_score'.format(name), ari, epoch)
        print("{} Clustering ACC: {}".format(name, cls_acc))

    # gather the stats from all processes
    # if name == "Val" and record:
    #     metric_logger.synchronize_between_processes()
    #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    #
    #     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # elif not record:
    #     return cls_acc
    # else:
    return None


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, student_output, teacher_output, label, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # print(self.ncrops)
        # print(len(student_out))
        # print(student_out[0].shape)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # print(teacher_out.shape)
        teacher_out = teacher_out.detach().chunk(2)

        total_dino_loss = 0
        total_ce_loss = 0
        n_dino_loss_terms = 0
        n_ce_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        for v in range(len(student_out)):
            for iq, q in enumerate(teacher_out):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_dino_loss += loss.mean()
                n_dino_loss_terms += 1
            total_ce_loss += self.ce_loss(student_out[v], label)
            n_ce_loss_terms += 1
        total_ce_loss /= n_ce_loss_terms
        total_dino_loss /= n_dino_loss_terms
        self.update_center(teacher_output)
        return total_dino_loss, total_ce_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
