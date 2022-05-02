# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss, SupCluLoss
import utils
import diffdist

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    # y_one_hot = y_one_hot.view(y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot

def kl_normal(qm, qv, pm, pv, yh):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return torch.mean(kl)

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


def train_one_epoch(model: torch.nn.Module, one_hot_layer, wce, wclu,
                    criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, logger=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion1 = torch.nn.CrossEntropyLoss()
    # criterion2 = SupCluLoss(temperature=0.07)
    criterion2 = torch.nn.MSELoss(reduction='mean')
    niters_per_epoch = len(data_loader)

    idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        idx += 1
        samples = samples.to(device, non_blocking=True)
        # print(samples.shape)
        targets = targets.to(device, non_blocking=True)
        # print(targets)
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        # print(samples.shape)
        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = critserion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            # print(samples.shape)
            outputs = model(samples)
            # fea = model.module.forward_features(samples)

            ce_loss = criterion1(outputs, targets)
            # target_en = torch.nn.functional.one_hot(targets,100)
            # print(target_en.type())
            # target_en = target_en.to(torch.half)
            # latent = one_hot_layer(target_en)
            # print(fea.shape)
            # print(latent.shape)
            # l2_loss = criterion2(fea,latent)


        ce_loss_value = ce_loss.item()
        # clu_loss_value = clu_loss.item()

        # loss = wce * ce_loss + wclu * l2_loss
        loss = wce * ce_loss

        if not math.isfinite(ce_loss_value):
            print("Loss is {}, stopping training".format(ce_loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=ce_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)
        # logger.add_scalar('Train_Loss/l2_loss', l2_loss, epoch * niters_per_epoch + idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_byol(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, logger=None, use_momentum=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # criterion1 = torch.nn.CrossEntropyLoss()
    # # criterion2 = SupCluLoss(temperature=0.07)
    # criterion2 = torch.nn.MSELoss(reduction='mean')
    niters_per_epoch = len(data_loader)

    idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        idx += 1
        samples = samples.to(device, non_blocking=True)
        # print(samples.shape)
        targets = targets.to(device, non_blocking=True)
        # print(targets)
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        # print(samples.shape)
        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = critserion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            # print(samples.shape)
            loss = model(samples)
            # fea = model.module.forward_features(samples)

            # ce_loss = criterion1(outputs, targets)
            # target_en = torch.nn.functional.one_hot(targets,100)
            # print(target_en.type())
            # target_en = target_en.to(torch.half)
            # latent = one_hot_layer(target_en)
            # print(fea.shape)
            # print(latent.shape)
            # l2_loss = criterion2(fea,latent)


        # ce_loss_value = ce_loss.item()
        # clu_loss_value = clu_loss.item()

        # loss = wce * ce_loss + wclu * l2_loss
        # loss = wce * ce_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if use_momentum:
            model.module.update_moving_average()
        logger.add_scalar('Train_Loss/loss', loss, epoch * niters_per_epoch + idx)
        # logger.add_scalar('Train_Loss/l2_loss', l2_loss, epoch * niters_per_epoch + idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_dgrl(model: torch.nn.Module, one_hot_layer, wce, wclu,
                    criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, logger=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion1 = torch.nn.CrossEntropyLoss()
    # criterion2 = SupCluLoss(temperature=0.07)
    criterion2 = torch.nn.MSELoss(reduction='mean')
    niters_per_epoch = len(data_loader)

    idx = 0
    class_label = torch.Tensor(np.array(range(100)))
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        idx += 1
        samples = samples.to(device, non_blocking=True)
        # print(samples.shape)
        targets = targets.to(device, non_blocking=True)
        center_labels_var = torch.autograd.Variable(class_label.to(torch.long)).cuda()
        labels_var_one_hot = to_one_hot(targets, n_dims=100)
        # print(targets)
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        # print(samples.shape)
        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = critserion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            # print(samples.shape)
            outputs = model(samples)
            fea = model.module.forward(samples)
            class_weight = model.module.head.weight
            fea = fea - 4 * labels_var_one_hot.cuda()
            ce_loss = criterion1(fea, targets)
            center_loss = criterion1(torch.mm(class_weight, torch.t(class_weight)), center_labels_var)


        ce_loss_value = ce_loss.item()
        # clu_loss_value = clu_loss.item()

        loss = ce_loss + 0.5 * center_loss

        # if not math.isfinite(ce_loss_value):
        #     print("Loss is {}, stopping training".format(ce_loss_value))
        #     sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=ce_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)
        logger.add_scalar('Train_Loss/center_loss', center_loss, epoch * niters_per_epoch + idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_aux(model: torch.nn.Module, one_hot_layer, wce, wclu,
                    criterion: DistillationLoss,
                    train_data_loader: Iterable, aux_data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, logger=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss(reduction='mean')
    niters_per_epoch = len(train_data_loader)

    idx = 0
    for samples, targets in metric_logger.log_every(train_data_loader, print_freq, header):
        idx += 1
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=not fp32):

            outputs = model(samples)
            # fea = model.module.forward_features(samples)

            ce_loss = criterion1(outputs, targets)
            # target_en = torch.nn.functional.one_hot(targets,100)
            # print(target_en.type())
            # target_en = target_en.to(torch.half)
            # latent = one_hot_layer(target_en)
            # print(fea.shape)
            # print(latent.shape)
            # l2_loss = criterion2(fea,latent)


        ce_loss_value = ce_loss.item()
        # clu_loss_value = clu_loss.item()

        loss = wce * ce_loss

        if not math.isfinite(ce_loss_value):
            print("Loss is {}, stopping training".format(ce_loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=ce_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)
        # logger.add_scalar('Train_Loss/l2_loss', l2_loss, epoch * niters_per_epoch + idx)

    niters_per_epoch = len(aux_data_loader)
    for samples, targets in metric_logger.log_every(aux_data_loader, print_freq, header):
        idx += 1
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=not fp32):

            # outputs = model(samples)
            fea = model.module.forward_features(samples)
            # fea = model(samples)

            # ce_loss = criterion1(outputs, targets)
            target_en = torch.nn.functional.one_hot(targets,100)
            # print(target_en.type())
            target_en = target_en.to(torch.half)
            latent = one_hot_layer(target_en)
            # print(fea.shape)
            # print(latent.shape)
            l2_loss = criterion2(fea,latent)

        # ce_loss_value = ce_loss.item()
        # clu_loss_value = clu_loss.item()
        l2_loss_value = l2_loss.item()

        loss = wclu * l2_loss

        if not math.isfinite(l2_loss_value):
            print("Loss is {}, stopping training".format(ce_loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # metric_logger.update(loss=l2_loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # logger.add_scalar('Train_Loss/ce_loss', ce_loss, epoch * niters_per_epoch + idx)
        logger.add_scalar('Train_Loss/l2_loss', l2_loss, epoch * niters_per_epoch + idx)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def batch_gather_ddp(images):
    """
    gather images from different gpus and shuffle between them
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    images_gather = []
    for i in range(4):
        batch_size_this = images[i].shape[0]
        # print(images[i].shape)
        images_gather.append(concat_all_gather(images[i]))
        batch_size_all = images_gather[i].shape[0]
    num_gpus = batch_size_all // batch_size_this

    n, c, h, w = images_gather[0].shape
    permute = torch.randperm(n * 4).cuda()
    torch.distributed.broadcast(permute, src=0)
    images_gather = torch.cat(images_gather, dim=0)
    images_gather = images_gather[permute, :, :, :]
    col1 = torch.cat([images_gather[0:n], images_gather[n:2 * n]], dim=3)
    col2 = torch.cat([images_gather[2 * n:3 * n], images_gather[3 * n:]], dim=3)
    images_gather = torch.cat([col1, col2], dim=2)

    # bs = images_gather.shape[0] // num_gpus
    # gpu_idx = torch.distributed.get_rank()

    return images_gather, permute, n

def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    tensors_gather = diffdist.functional.all_gather(tensors_gather, tensor, next_backprop=None, inplace=True)

    output = torch.cat(tensors_gather, dim=0)
    return output

def train_one_epoch_jigsaw(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False,
                    clu_fc = None,
                    loc_fc = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    criterion_clu = SupCluLoss(temperature=0.07)
    criterion_loc = torch.nn.CrossEntropyLoss()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(samples[0].shape)
        for i in range(4):
            samples[i] = samples[i].to(device, non_blocking=True)

        images_gather, permute, bs_all = batch_gather_ddp(samples)
        # print(len(images_gather))
        # print(permute)
        # print("Finished permutation!!!")
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = criterion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            fea = model.module.forward_features(images_gather)

            # fea = fea.contiguous()
            # print(fea.shape)
            # q_gather = concat_all_gather(fea)
            # # print(q_gather.shape)
            # n, c, h, w = q_gather.shape
            # c1, c2 = q_gather.split([1, 1], dim=2)
            # f1, f2 = c1.split([1, 1], dim=3)
            # f3, f4 = c2.split([1, 1], dim=3)
            # q_gather = torch.cat([f1, f2, f3, f4], dim=0)
            # q_gather = q_gather.view(n * 4, -1)
            fea_len = fea.shape[1]
            # print(fea.shape)
            f1, f2, f3, f4 = fea.split([fea_len//4,fea_len//4,fea_len//4,fea_len//4],dim=1)
            q_gather = torch.cat([f1, f2, f3, f4], dim=0)
            # print(q_gather.shape)
            # # clustering branch
            label_clu = permute % bs_all
            q_clu = clu_fc(q_gather)
            q_clu = torch.nn.functional.normalize(q_clu, dim=1)
            loss_clu = criterion_clu(q_clu, label_clu)

            label_loc = torch.LongTensor([0] * bs_all + [1] * bs_all + [2] * bs_all + [3] * bs_all).cuda()
            label_loc = label_loc[permute]
            q_loc = loc_fc(q_gather)
            loss_loc = criterion_loc(q_loc, label_loc)
            # outputs = model(samples)
            # loss = criterion(samples, outputs, targets)
            loss = loss_clu + loss_loc

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, epoch, logger=None, name="Val", record=True):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = SupCluLoss(temperature=0.07)
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
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            fea = model.module.forward_features(images)
            fea = model(images)
            ce_loss = criterion1(outputs, targets)
            # clu_loss = criterion2(fea, targets)
            # ce_loss = criterion1(samples, outputs, targets)
            # clu_loss = criterion2(samples, outputs, targets)
            # loss = criterion(output, target)
        if record:
            logger.add_scalar('{}/ce_loss'.format(name), ce_loss, epoch * niters_per_epoch + idx)
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

        if name == "Val" and record:
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=ce_loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    kmeans = KMeans(n_clusters=len(set(target_list)), random_state=1).fit(fea_list)
    # print(target_list.shape)
    # print(len(kmeans.labels_))
    # target_list = target_list.
    cls_acc = cluster_acc(target_list, kmeans.labels_)
    nmi = nmi_score(target_list, kmeans.labels_)
    ari = ari_score(target_list, kmeans.labels_)
    if record:
        logger.add_scalar('{}/cls_acc'.format(name), cls_acc, epoch)
        logger.add_scalar('{}/nmi_score'.format(name), nmi, epoch)
        logger.add_scalar('{}/ari_score'.format(name), ari, epoch)
        print("{} Clustering ACC: {}".format(name, cls_acc))

    # gather the stats from all processes
    if name == "Val" and record:
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    elif not record:
        return cls_acc
    else:
        return None

@torch.no_grad()
def evaluate_byol(data_loader, model, device, epoch, logger=None, name="Val", record=True):
    # criterion1 = torch.nn.CrossEntropyLoss()
    # criterion2 = SupCluLoss(temperature=0.07)
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
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            _, fea = model(images, return_embedding=True)
            # fea = model.module.forward_features(images)
            # ce_loss = criterion1(outputs, targets)
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
    nmi = nmi_score(target_list, kmeans.labels_)
    ari = ari_score(target_list, kmeans.labels_)
    if record:
        logger.add_scalar('{}/cls_acc'.format(name), cls_acc, epoch)
        logger.add_scalar('{}/nmi_score'.format(name), nmi, epoch)
        logger.add_scalar('{}/ari_score'.format(name), ari, epoch)
        print("{} Clustering ACC: {}".format(name, cls_acc))

    # gather the stats from all processes
    if name == "Val" and record:
        metric_logger.synchronize_between_processes()
        # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        return None
    elif not record:
        return cls_acc
    else:
        return None