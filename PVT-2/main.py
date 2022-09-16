# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import sys
import os

from pathlib import Path

import timm
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from tensorboardX import SummaryWriter
import logging
from data.data.cub import CustomCub2011, get_cub_datasets
from data.data.nabird import CustomNABirdV1, get_nabird_datasets
from data.data.stanford_cars import CarsDataset, get_scar_datasets
from data.data.cifar import get_cifar10_datasets, get_cifar100_datasets
from data.data.nabird import get_nabird_datasets
from data.data.tinyimagenet import get_tiny_image_net_datasets
from data.utils import TransformTwice, RandomTranslateWithReflect
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import train_one_epoch, update_means, evaluate
from losses import DistillationLoss
from samplers import RASampler
# import models
# import pvt
# import pvt_v2
import vision_transformer as vits
import utils
import collections
# from vits import vit_small_single
# from timm.models.vision_transformer import VisionTransformer, _cfg
import pickle

from dec import DEC
# from model_aevit import ae_vit_base_patch16_dec512d8b as ae_vit_base


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--num_train_classes', default=100, type=int)
    parser.add_argument('--num_aux_classes', default=50, type=int)
    parser.add_argument('--num_test_classes', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # parser.add_argument('--model-ema', action='store_true')
    # parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    # parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    # parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--wclu', type=float, default=1)
    parser.add_argument('--wce', type=float, default=1)
    parser.add_argument('--patch_size', type=int, default=16)

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--ce_warmup_epoch', default=20, type=int)

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    # parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
    #                     help='Name of teacher model to train (default: "regnety_160"')
    # parser.add_argument('--teacher-path', type=str, default='')
    # parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    # parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    # parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='CUB', choices=['CIFAR10', 'CIFAR100', 'TIMNT', 'IMNET', 'INAT', 'INAT19', 'CUB', 'SCAR', 'NABird'],
                        type=str, help='dataset path')
    parser.add_argument('--use-mcloader', action='store_true', default=False, help='Use mcloader')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # avail_pretrained_models = timm.list_models(pretrained=True)
    # for i in avail_pretrained_models:
    #     print(i)
    # sys.exit(0)
    utils.init_distributed_mode(args)
    print(args)
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    np.random.seed(args.seed)
    # random.seed(seed)

    cudnn.benchmark = True

    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'CUB':
        args.nb_classes = 100
        args.input_size = 224
        datasets = get_cub_datasets(train_transform, test_transform, seed=args.seed, num_train_classes=args.num_train_classes,
                                    num_auxiliary_classes=args.num_aux_classes, num_open_set_classes=args.num_test_classes)
    elif args.dataset == 'NABird':
        args.nb_classes = 304
        args.input_size = 224
        if args.num_train_classes is not None:
            datasets = get_nabird_datasets(train_transform, test_transform, seed=args.seed,
                                           num_train_classes=args.num_train_classes,
                                           num_auxiliary_classes=args.num_aux_classes,
                                           num_open_set_classes=args.num_test_classes)
            args.nb_classes = args.num_train_classes
        else:
            datasets = get_nabird_datasets(train_transform.test_transform, seed=args.seed)
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

    # mix_train_loader = DataLoader(datasets['mix_train'], batch_size=args.batch_size,
    #                               shuffle=True, sampler=None, num_workers=args.num_workers)
    # mix_test_loader = DataLoader(datasets['mix_test'], batch_size=args.batch_size,
    #                              shuffle=False, sampler=None, num_workers=args.num_workers)
    labeled_train_loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                                      shuffle=True, sampler=None, num_workers=args.num_workers)
    labeled_eval_loader = DataLoader(datasets['val'], batch_size=args.batch_size,
                                     shuffle=False, sampler=None, num_workers=args.num_workers)
    aux_test_loader = DataLoader(datasets['aux_test'], batch_size=args.batch_size,
                                 shuffle=False, sampler=None, num_workers=args.num_workers)
    test_unknown_loader = DataLoader(datasets['test_unknown'], batch_size=args.batch_size,
                                     shuffle=False, sampler=None, num_workers=args.num_workers)
    test_known_loader = DataLoader(datasets['test_known'], batch_size=args.batch_size,
                                   shuffle=False, sampler=None, num_workers=args.num_workers)
    data_loader_train = labeled_train_loader
    data_loader_val = labeled_eval_loader

    # print("1111111111111111111111111111111111111111111111")
    # args.wclu = 1
    # print(args.wclu)
    #Logger
    args.output_dir = os.path.join(args.output_dir,args.dataset, str(args.wce)+"_"+str(args.wclu)+"_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    logger = SummaryWriter(args.output_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(args))


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    # print(timm.list_models())
    model = create_model(
        args.model,
        # pretrained=False,
        pretrained=True,
        num_classes=args.nb_classes,
        # num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        )

    # model = DEC(args.num_train_classes, 768, model, 1.0)

    # model = ae_vit_base()
    # model.load_pretrained(
    #     "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz")

    # utils.save_on_master(model.state_dict(), "./imagenet_pretrained.pth")
    # print("Imagenet supervised pretrained weights saved!")
    # exit()

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("The model at {} has been loaded".format(args.finetune))
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # for BYOL and SimSiam
        # for k in list(checkpoint_model.keys()):
        #     if k.startswith("net."):
        #         checkpoint_model[k.replace("net.","")] = checkpoint_model[k]
        #         del checkpoint_model[k]
        #
        # for DINO
        # print(checkpoint_model.keys())
        # if "checkpoint" in args.finetune:
        if "teacher" in list(checkpoint_model.keys()):
            checkpoint_model = checkpoint_model['teacher']
            for k in list(checkpoint_model.keys()):
                if k.startswith("backbone."):
                    checkpoint_model[k.replace("backbone.","")] = checkpoint_model[k]
                    del checkpoint_model[k]

        # checkpoint_model = checkpoint_model['student']
        # for k in list(checkpoint_model.keys()):
        #     if k.startswith("module.backbone."):
        #         checkpoint_model[k.replace("module.backbone.", "")] = checkpoint_model[k]
        #         del checkpoint_model[k]

        # for MOCO_V3
        # checkpoint_model = checkpoint_model['state_dict']
        # # print(checkpoint_model.keys())
        # for k in list(checkpoint_model.keys()):
        #     # if k.startswith("module.base_encoder."):
        #     if k.startswith("module.encoder_q."):
        #         # checkpoint_model[k.replace("module.base_encoder.", "")] = checkpoint_model[k]
        #         checkpoint_model[k.replace("module.encoder_q.", "")] = checkpoint_model[k]
        #         del checkpoint_model[k]
        # print(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):

            if k in list(state_dict.keys()):
                print(k)
        model.load_state_dict(checkpoint_model, strict=False)

        print("Following variables are not in the pretrained weights")
        for k in list(state_dict.keys()):
            if k not in list(checkpoint_model.keys()):
                print(k)

        print("Loading the pretrained model")
    model.to(device)
    # one_hot_layer = torch.nn.Linear(100, 768)
    # one_hot_layer = one_hot_layer.cuda()

    model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEma(
    #         model,
    #         decay=args.model_ema_decay,
    #         device='cpu' if args.model_ema_force_cpu else '',
    #         resume='')

    model_without_ddp = model
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = SupCluLoss(temperature=0.07)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            msg = model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            msg = model_without_ddp.load_state_dict(checkpoint)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.model_ema:
            #     utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        # model.module.reset_classifier(0)
        model.eval()
        cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], use_cuda=True)


        print("Start evaluation")
        open("{}/tk_result.txt".format(args.output_dir),"w").close()
        open("{}/tk_fea.txt".format(args.output_dir), "w").close()
        open("{}/tk_target.txt".format(args.output_dir),"w").close()
        open("{}/tu_result.txt".format(args.output_dir), "w").close()
        open("{}/tu_fea.txt".format(args.output_dir), "w").close()
        open("{}/tu_target.txt".format(args.output_dir), "w").close()
        open("{}/aux_result.txt".format(args.output_dir), "w").close()
        open("{}/aux_fea.txt".format(args.output_dir), "w").close()
        open("{}/aux_target.txt".format(args.output_dir), "w").close()

        for batch_idx, (images, target) in enumerate(tqdm(test_known_loader)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = torch.Tensor.cpu(target).detach().numpy()
            with torch.cuda.amp.autocast():
                output = model(images)
                fea = model.module.forward_features(images)
                output = torch.Tensor.cpu(output).detach().numpy()
                fea = torch.Tensor.cpu(fea).detach().numpy()
                # print(output.shape)
            with open('{}/tk_result.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, output, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/tk_fea.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/tk_target.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, target, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')


        for batch_idx, (images, target) in enumerate(tqdm(test_unknown_loader)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = torch.Tensor.cpu(target).detach().numpy()
            with torch.cuda.amp.autocast():
                output = model(images)
                fea = model.module.forward_features(images)
                output = torch.Tensor.cpu(output).detach().numpy()
                fea = torch.Tensor.cpu(fea).detach().numpy()
            with open('{}/tu_result.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, output, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/tu_fea.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/tu_target.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, target, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
        for batch_idx, (images, target) in enumerate(tqdm(aux_test_loader)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = torch.Tensor.cpu(target).detach().numpy()
            with torch.cuda.amp.autocast():
                output = model(images)
                fea = model.module.forward_features(images)
                output = torch.Tensor.cpu(output).detach().numpy()
                fea = torch.Tensor.cpu(fea).detach().numpy()
            with open('{}/aux_result.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, output, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/aux_fea.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, fea, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')
            with open('{}/aux_target.txt'.format(args.output_dir), 'ab') as f:
                np.savetxt(f, target, fmt='%f', delimiter=' ', newline='\r')
                f.write(b'\n')

        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_val_loss = 99999999

    # means = update_means(model, data_loader_train, args=args, logger=logger, epoch=-1)
    means = None
    evaluate(test_known_loader, labeled_eval_loader, aux_test_loader, test_unknown_loader, model, device, -1, args,
             logger, means)
    for epoch in range(args.start_epoch, args.epochs):
        if args.fp32_resume and epoch > args.start_epoch + 1:
            args.fp32_resume = False
        loss_scaler._scaler = torch.cuda.amp.GradScaler(enabled=not args.fp32_resume)

        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        if epoch < args.ce_warmup_epoch:
            train_stats = train_one_epoch(
                model, args.wce, args.wclu, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                fp32=args.fp32_resume, logger=logger, use_clus=False, aux_set=False, means=means, args=args
            )
            # print("One epoch training done!")
        else:
            args.wce = 0
            train_stats = train_one_epoch(
                model, args.wce, args.wclu, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
                fp32=args.fp32_resume, logger=logger, use_clus=True, aux_set=False, means=means, args=args
            )

        lr_scheduler.step(epoch)
        # print("One epoch training done!")

        # means = update_means(model, data_loader_train, args=args, logger=logger, epoch=epoch)

        val_loss  = evaluate(test_known_loader, labeled_eval_loader, aux_test_loader, test_unknown_loader, model, device, epoch, args, logger, means)
        # print("Evaluation finished!")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.output_dir:
                print("Save model weights of current epoch!")
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)


        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # print(args)
    args = utils.update_from_config(args)
    # print(args)
    # args.lr = 0.0005
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
