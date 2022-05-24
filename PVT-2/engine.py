# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import matplotlib

import torch
from torchvision.utils import make_grid
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
from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix
import seaborn as sns
# from scipy.optimize import linear_sum_assignment as linear_assignment

from dec import target_distribution

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

def update_means(model, train_data_loader, aux_data_loader=None, args=None, logger=None, epoch=None):
    fea_list = None
    target_list = None
    with torch.no_grad():
        for it, (images, label) in enumerate(train_data_loader):
            images = images.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # fea = student.module.forward_features(images)
            # fea = model.module.forward_features(images)
            # fea = model.module.encoder.forward_features(images)
            _, _, fea, _ = model(images)
            # fea = torch.nn.functional.normalize(fea, p=2)
            feas = torch.Tensor.cpu(fea).detach().numpy()
            # print(feas.shape)
            if fea_list is None:
                fea_list = feas
            else:
                fea_list = np.vstack((fea_list, feas))
            targets_temp = torch.Tensor.cpu(label).detach().numpy()
            if target_list is None:
                target_list = targets_temp
            else:
                target_list = np.concatenate((target_list, targets_temp))

        if aux_data_loader is not None:
            for it, (images, label) in enumerate(aux_data_loader):
                # move images to gpu
                images = [im.cuda(non_blocking=True) for im in images]
                label = label.cuda(non_blocking=True)

                # fea = student.module.forward_features(images)
                _, fea = model(images)
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

    target_list = target_list.reshape(-1)
    # print(set(target_list))
    target_list = np.expand_dims(target_list, axis=1)
    # print(target_list.shape)
    # print(fea_list.shape)
    fea_tar = np.hstack((target_list, fea_list)).squeeze()
    means_dict = {}
    for i in np.unique(fea_tar[:, 0]):
        tmp = fea_list[np.where(fea_tar[:, 0] == i)]
        means_dict[i] = np.mean(tmp, axis=0)
    tmp_means = []
    if aux_data_loader is not None:
        for k in range(args.num_train_classes + args.num_aux_classes):
            tmp_means.append(means_dict[k])
    else:
        for k in range(args.num_train_classes):
            tmp_means.append(means_dict[k])
    means = torch.as_tensor(np.array(tmp_means)).cuda()
    # print(means)
    print("The means are updated")
    print("There are {} classes".format(means.shape[0]))
    dm = distance_matrix(np.array(tmp_means), np.array(tmp_means))
    # fig = plt.figure()
    # plt.imshow(dm, cmap='hot', interpolation='nearest')
    # logger.add_figure("distance_heatmap", fig, epoch)

    ax = sns.heatmap(dm, linewidth=0, cmap='coolwarm')
    fig = ax.get_figure()
    logger.add_figure("distance_heatmap", fig, epoch)

    return means

def train_one_epoch(model: torch.nn.Module, wce, wclu,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, logger=None, use_clus=False, aux_set=False, means=None, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    criterion1 = torch.nn.CrossEntropyLoss()
    # criterion2 = torch.nn.MSELoss(reduction='mean')
    loss_function = torch.nn.KLDivLoss(reduction='sum')
    niters_per_epoch = len(data_loader)

    means = torch.diag(torch.Tensor([10 for i in range(args.num_train_classes)])).cuda()

    idx = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        idx += 1
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=not fp32):

            re_loss, pred, latent, rec = model(images)
            ce_loss = criterion1(pred, targets)

        loss = wce * ce_loss + re_loss
        # print(ce_loss)
        # print("1111111")
        # print(re_loss)
        ce_loss_value = ce_loss.item()

        if use_clus:
            # fea = model.module.forward_features(images)
            # fea = model.module.encoder.forward_features(images)
            # fea = torch.nn.functional.normalize(fea, p=2)
            fea = pred
            if aux_set:
                targets_temp = torch.Tensor.cpu(targets).detach().numpy() + args.num_train_classes
            else:
                targets_temp = torch.Tensor.cpu(targets).detach().numpy()
            targets_temp = targets_temp.reshape(-1)

            # fea = fea.unsqueeze(1).repeat_interleave(args.num_train_classes + args.num_aux_classes, dim=1)
            # fea = fea.unsqueeze(1).repeat_interleave(args.num_train_classes, dim=1)
            #
            # dist = torch.sqrt(torch.sum(torch.square(fea - means), dim=-1))
            # # mask = torch.full((fea.shape[0], args.num_train_classes + args.num_aux_classes), -1, dtype=int).cuda()
            # pos_mask = torch.full((fea.shape[0], args.num_train_classes), 0, dtype=int).cuda()
            # neg_mask = torch.full((fea.shape[0], args.num_train_classes), 1, dtype=int).cuda()
            # for id, index in enumerate(targets_temp):
            #     # mask[id, index] = args.num_train_classes
            #     pos_mask[id, index] = 1
            #     neg_mask[id, index] = 1
            #
            # clu_loss = torch.mean(torch.sum(torch.mul(dist, pos_mask), dim=1)) + torch.mean(
            #     torch.sum(torch.mul(1 / dist, pos_mask), dim=1))
            # print(mask)
            #
            # # DCN loss
            # clu_loss = torch.tensor(0.).cuda()
            # for i in range(args.batch_size):
            #     diff_vec = fea[i] - means[targets_temp[i]]
            #     sample_dist_loss = torch.matmul(diff_vec.view(1,-1),
            #                                     diff_vec.view(-1,1))
            #     clu_loss += 0.5 * torch.squeeze(sample_dist_loss)

            # CAC loss
            # distance matrix from feature to class anchors
            fea = fea.unsqueeze(1).expand(-1, args.num_train_classes, -1)
            anchors = means.unsqueeze(0).expand(fea.shape[0],-1,-1)
            dist = torch.norm(fea - anchors, 2, 2)
            true = torch.gather(dist, 1, targets.view(-1,1)).view(-1)
            non_gt = torch.Tensor([[i for i in range(args.num_train_classes) if targets[x] !=i] for x in range(len(dist))]).long().cuda()
            others = torch.gather(dist, 1, non_gt)
            anchor = torch.mean(true)
            tuplet = torch.exp(-others+true.unsqueeze(1))
            tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim=1)))

            logger.add_scalar('Train_Loss/anchor_loss', anchor, epoch * niters_per_epoch + idx)
            logger.add_scalar('Train_Loss/tuplet_loss', tuplet, epoch * niters_per_epoch + idx)
            loss += 0.1 * anchor + tuplet

            # DEC loss
            # target_p = target_distribution(fea)
            # clu_loss = loss_function(fea.log(), target_p) / fea.shape[0]


            # logger.add_scalar('Train_Loss/clu_loss', clu_loss, epoch * niters_per_epoch + idx)
            # loss += wclu * clu_loss


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
        logger.add_scalar('Train_Loss/re_loss', re_loss, epoch * niters_per_epoch + idx)
        if idx == 5:
            # print(rec.shape)
            img_grid = make_grid(rec, nrow=4)
            logger.add_image('Train/reconstructed', img_grid,epoch)
            img_grid = make_grid(images, nrow=4)
            logger.add_image('Train/original', img_grid, epoch)
        # logger.add_scalar('Train_Loss/l2_loss', l2_loss, epoch * niters_per_epoch + idx)

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

@torch.no_grad()
def evaluate(known_data_loader, val_data_loader, aux_data_loader, unknown_data_loader, model, device, epoch, args, logger=None, means=None):

    # switch to evaluation mode
    model.eval()

    criterion1 = torch.nn.CrossEntropyLoss()
    fea_list = None
    target_list = None
    # means = torch.from_numpy(means)

    val_clu_loss = 0
    val_ce_loss = 0
    val_re_loss = 0
    # loss_function = torch.nn.KLDivLoss(reduction='sum')
    for images, targets in val_data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # outputs = model(images)
            re_loss, pred, fea, rec = model(images)
            # fea = model.module.forward_features(images)
            # fea = model.module.encoder.forward_features(images)
            # fea = torch.nn.functional.normalize(fea, p=2)
            ce_loss = criterion1(pred, targets)
            val_re_loss += re_loss
            val_ce_loss += ce_loss

            targets_temp = torch.Tensor.cpu(targets).detach().numpy()
            targets_temp = targets_temp.reshape(-1)

            # target_p = target_distribution(fea)
            # clu_loss = loss_function(fea.log(), target_p) / fea.shape[0]

            clu_loss = torch.tensor(0.).cuda()
            for i in range(fea.shape[0]):
                diff_vec = fea[i] - means[targets_temp[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                                diff_vec.view(-1, 1))
                clu_loss += 0.5 * torch.squeeze(sample_dist_loss)

            # fea_mod = fea.unsqueeze(1).repeat_interleave(args.num_train_classes, dim=1)
            # # print(fea.shape)
            # # print(means.shape)
            # dist = torch.sqrt(torch.sum(torch.square(fea_mod - means), dim=-1))
            # # dist = torch.nn.functional.normalize(dist)
            # # print(dist.shape)
            # pos_mask = torch.full((fea.shape[0], args.num_train_classes), 0, dtype=int).cuda()
            # neg_mask = torch.full((fea.shape[0], args.num_train_classes), 1, dtype=int).cuda()
            # for id, index in enumerate(targets_temp):
            #     # mask[id, index] = args.num_train_classes
            #     pos_mask[id, index] = 1
            #     neg_mask[id, index] = 1
            #
            # clu_loss = torch.mean(torch.sum(torch.mul(dist, pos_mask), dim=1)) + torch.mean(
            #     torch.sum(torch.mul(1 / dist, pos_mask), dim=1))
            val_clu_loss += clu_loss

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
    # print(fea_list.shape)
    kmeans = KMeans(n_clusters=args.num_train_classes, random_state=1).fit(fea_list)
    val_cls_acc = cluster_acc(target_list, kmeans.labels_)
    logger.add_scalar('Val/ce_loss', val_ce_loss, epoch)
    logger.add_scalar('Val/re_loss', val_re_loss, epoch)
    logger.add_scalar('Val/clu_loss', val_clu_loss, epoch)
    logger.add_scalar('{}/cls_acc'.format("Val"), val_cls_acc, epoch)

    if epoch % 5 == 0:
        fea_list = None
        target_list = None

        for images, targets in known_data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                # fea = model.module.forward_features(images)
                re_loss, pred, fea, rec = model(images)
                # fea = model.module.encoder.forward_features(images)
                # fea = torch.nn.functional.normalize(fea, p=2)

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

        for images, targets in aux_data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                # fea = model.module.forward_features(images)
                re_loss, pred, fea, rec = model(images)
                # fea = model.module.encoder.forward_features(images)
                # fea = torch.nn.functional.normalize(fea, p=2)

            feas = torch.Tensor.cpu(fea).detach().numpy()
            fea_list = np.vstack((fea_list,feas))
            targets_temp = torch.Tensor.cpu(targets).detach().numpy() + args.num_train_classes
            target_list = np.concatenate((target_list, targets_temp))

        known_set_len = target_list.shape[0]

        for images, targets in unknown_data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                # fea = model.module.forward_features(images)
                re_loss, pred, fea, rec = model(images)
                # fea = model.module.encoder.forward_features(images)
                # fea = torch.nn.functional.normalize(fea, p=2)

            feas = torch.Tensor.cpu(fea).detach().numpy()
            fea_list = np.vstack((fea_list,feas))
            targets_temp = torch.Tensor.cpu(targets).detach().numpy() + args.num_train_classes + args.num_aux_classes
            target_list = np.concatenate((target_list, targets_temp))

        kmeans = KMeans(n_clusters=args.num_train_classes, random_state=1).fit(fea_list[:known_set_len])
        cls_acc = cluster_acc(target_list[:known_set_len], kmeans.labels_)
        logger.add_scalar('{}/cls_acc'.format("Known"), cls_acc, epoch)
        print("{} Clustering ACC: {}".format("Known", cls_acc))

        kmeans = KMeans(n_clusters=args.num_aux_classes+args.num_test_classes, random_state=1).fit(fea_list[known_set_len:])
        cls_acc = cluster_acc(target_list[known_set_len:], kmeans.labels_)
        logger.add_scalar('{}/cls_acc'.format("Unknown"), cls_acc, epoch)
        print("{} Clustering ACC: {}".format("Unknown", cls_acc))

        kmeans = KMeans(n_clusters=args.num_train_classes+args.num_aux_classes+args.num_test_classes, random_state=1).fit(fea_list)
        cls_acc = cluster_acc(target_list, kmeans.labels_)
        logger.add_scalar('{}/cls_acc'.format("All"), cls_acc, epoch)
        print("{} Clustering ACC: {}".format("All", cls_acc))

        tsne = TSNE(n_components=2)
        twodim_fea = tsne.fit_transform(fea_list)
        kmeans = KMeans(n_clusters=args.num_train_classes + args.num_aux_classes + args.num_test_classes,
                        random_state=1).fit(twodim_fea)
        cls_acc = cluster_acc(target_list, kmeans.labels_)
        logger.add_scalar('{}/tsne_cls_acc'.format("All"), cls_acc, epoch)
        print("{} TSNE Clustering ACC: {}".format("All", cls_acc))

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, args.num_train_classes + args.num_aux_classes + args.num_test_classes))
        color_list = [colors[i] for i in target_list.tolist()]
        fig = plt.figure()
        plt.scatter(twodim_fea[:, 0].tolist(), twodim_fea[:, 1].tolist(), s=2, c=color_list)
        logger.add_figure("fea_vis", fig, epoch)

        tsne = TSNE(n_components=2)
        twodim_fea = tsne.fit_transform(torch.Tensor.cpu(means).detach().numpy())
        colors = matplotlib.cm.rainbow(
            np.linspace(0, 1, args.num_train_classes))
        color_list = colors
        fig = plt.figure()
        plt.scatter(twodim_fea[:, 0].tolist(), twodim_fea[:, 1].tolist(), s=2, c=color_list)
        for i in range(args.num_train_classes):
            plt.annotate(str(i), (twodim_fea[i,0], twodim_fea[i,1]))
        logger.add_figure("fea_mean_vis", fig, epoch)
        # with open(args.output_dir + "/fea.txt","wb") as f:
        #     np.savetxt(f, fea_list, fmt='%f', delimiter=' ', newline='\r')
        #     f.write(b'\n')
        #
        # with open(args.output_dir + "/target.txt","wb") as f:
        #     np.savetxt(f, target_list, fmt='%f', delimiter=' ', newline='\r')
        #     f.write(b'\n')

    return args.wce * val_ce_loss + args.wclu * val_clu_loss

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