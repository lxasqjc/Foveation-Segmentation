# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
import pandas as pd
import numpy as np
# Numerical libs
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import entropy
# Our libs
from config import cfg
from dataset import TrainDataset, imresize, b_imresize #, ValDataset
from models import ModelBuilder, SegmentationModule, FovSegmentationModule
from utils import AverageMeter, parse_devices, setup_logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from eval import eval_during_train
from eval_multipro import eval_during_train_multipro
from criterion import OhemCrossEntropy, DiceCoeff, DiceLoss, FocalLoss



# train one epoch
def train(segmentation_module, iterator, optimizers, epoch, cfg, history=None, foveation_module=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)
    if cfg.MODEL.foveation:
        foveation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        if type(batch_data) is not list:
            single_gpu_mode = True
            batch_data['img_data'] = batch_data['img_data'][0].cuda()
            batch_data['seg_label'] = batch_data['seg_label'][0].cuda()
            batch_data = [batch_data]
        else:
            single_gpu_mode = False
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()
        if cfg.MODEL.foveation:
            foveation_module.zero_grad()

        # adjust learning rate non_foveation
        if not cfg.MODEL.foveation:
            cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
            adjust_learning_rate(optimizers, cur_iter, cfg)

        # Foveation
        if cfg.MODEL.foveation:
            # Note by sudo_ means here is only for size estimation purpose
            # because batch_data is obtained by user modified DataParallel, s.t. batch_data is a list with length as len(gpus)
            # and each batch_data[i] is the actualy dict(batch_data) returned in dataset.TrainDataset
            # for ib in range(len(batch_data)):
                # print('img_data shape: ',  batch_data[ib]['img_data'].shape)
            sudo_X, sudo_Y = batch_data[0]['img_data'], batch_data[0]['seg_label']
            fov_map_scale = cfg.MODEL.fov_map_scale
            # NOTE: although here we use batch imresize yet in practical batch size for X = 1
            sudo_X_lr = b_imresize(sudo_X, (round(sudo_X.shape[2]/fov_map_scale), round(sudo_X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')
            if cfg.TRAIN.auto_fov_location_step:
                cfg.TRAIN.fov_location_step = round(sudo_X.shape[2]/fov_map_scale)*round(sudo_X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))
            # foveation (crop as you go)
            fov_location_batch_step = 0
            if cfg.TRAIN.sync_location == 'rand':           # bp         at each step and sync at random
                rand_location = random.randint(1, cfg.TRAIN.fov_location_step-1)
            elif cfg.TRAIN.sync_location == 'mean_mbs':     # bp and opt at each step and sync at random (last of random X_lr_cord list) with average loss
                rand_location = cfg.TRAIN.fov_location_step
            elif cfg.TRAIN.sync_location == 'none_sync':    # bp and opt at each step
                rand_location = cfg.TRAIN.fov_location_step


            # mini_batch
            X_lr_cord = []
            for xi in range(sudo_X_lr.shape[2]):
                for yi in range(sudo_X_lr.shape[3]):
                    X_lr_cord.append((xi,yi))
            random.shuffle(X_lr_cord)
            mbs = cfg.TRAIN.mini_batch_size
            mb_iter_count = 0
            mb_idx = 0
            mb_idx_count = 0
            while mb_idx < len(X_lr_cord) and mb_idx_count < rand_location:
                # correct zero_grad https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                # https://discuss.pytorch.org/t/whats-the-difference-between-optimizer-zero-grad-vs-nn-module-zero-grad/59233
                segmentation_module.zero_grad()
                foveation_module.zero_grad()

                batch_iters = rand_location
                cur_iter = fov_location_batch_step + (i-1)*batch_iters + (epoch-1)*cfg.TRAIN.epoch_iters*batch_iters
                # print('original max_iter:', cfg.TRAIN.max_iters)
                if cfg.TRAIN.fov_scale_lr != '' or cfg.TRAIN.fov_scale_weight_decay != '':
                    # weighted patch size normalized _ mini_batch average
                    if mb_idx == 0:
                        wpsn_mb = 1
                    else:
                        wpsn_mb = wpsn_mb/mbs
                if cfg.TRAIN.sync_location != 'rand':
                    fov_max_iters = batch_iters * cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

                    if cfg.TRAIN.fov_scale_lr == 'pen_sp': # penalty small patch, smaller average patch size smaller learning rate
                        lr_scale=float(wpsn_mb)
                    elif cfg.TRAIN.fov_scale_lr == 'pen_lp': # penalty large patch, larger average patch size smaller learning rate
                        lr_scale=float(1-wpsn_mb)
                    else:
                        lr_scale = 1.
                    if cfg.TRAIN.fov_scale_weight_decay == 'reg_sp': # regularise small patch, smaller average patch size larger regularisation
                        wd_scale=float(1-wpsn_mb)
                    elif cfg.TRAIN.fov_scale_weight_decay == 'reg_lp': # regularise large patch, larger average patch size larger regularisation
                        wd_scale=float(wpsn_mb)
                    else:
                        wd_scale = 1.

                    if cfg.TRAIN.fov_scale_lr != '' or cfg.TRAIN.fov_scale_weight_decay != '':
                        wpsn_mb = 0

                    # print('before fov_pow lr_scale={}, wd_scale={}'.format(lr_scale, wd_scale))
                    adjust_learning_rate(optimizers, cur_iter, cfg, lr_mbs=True, f_max_iter=fov_max_iters, lr_scale=lr_scale, wd_scale=wd_scale)
                    if cfg.MODEL.gumbel_tau_anneal:
                        adjust_gms_tau(cur_iter, cfg, r=1./fov_max_iters)
                if cfg.TRAIN.entropy_regularisation:
                    mbs_mean_entropy_reg = 0
                xi = []
                yi = []
                mini_batch_sample = 0
                while mini_batch_sample < mbs and mb_idx < len(X_lr_cord):
                    xi.append(X_lr_cord[mb_idx][0])
                    yi.append(X_lr_cord[mb_idx][1])
                    mb_idx += 1
                    fov_location_batch_step += 1
                    mb_idx_count += 1
                    mini_batch_sample += 1
                xi = tuple(xi)
                yi = tuple(yi)

                for idx in range(len(batch_data)):
                    batch_data[idx]['cor_info'] = (xi, yi, rand_location, fov_location_batch_step)
                if fov_location_batch_step == rand_location:
                    if single_gpu_mode:
                        patch_data, F_Xlr, print_grad = foveation_module(batch_data[0])
                    else:
                        patch_data, F_Xlr, print_grad = foveation_module(batch_data)
                else:
                    if single_gpu_mode:
                        patch_data, F_Xlr = foveation_module(batch_data[0])
                    else:
                        patch_data, F_Xlr = foveation_module(batch_data)

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
                # by set base = len(patch_bank), uniform distribution will have entropy = 1 (so absolute uncertain)
                if cfg.TRAIN.entropy_regularisation:
                    # comprosed solution consider batch size != 1
                    F_Xlr_c = F_Xlr.clone()
                    if cfg.MODEL.gumbel_softmax:
                        F_Xlr_c = F_Xlr_c.exp()

                    mean_entropy_reg = 0
                    for i_batch in range(F_Xlr_c.shape[0]):
                        mean_entropy_reg += (-F_Xlr_c[i_batch,:,xi,yi]*F_Xlr_c[i_batch,:,xi,yi].log()).sum()
                    mbs_mean_entropy_reg += mean_entropy_reg/(rand_location//mbs)

                if cfg.TRAIN.entropy_regularisation:
                    # comprosed solution consider batch size != 1
                    mean_entropy = 0
                    for i_batch in range(F_Xlr.shape[0]):
                        mean_entropy += (entropy(F_Xlr[i_batch,:,xi,yi].cpu().detach().numpy(), base=len(cfg.MODEL.patch_bank)).mean())/F_Xlr.shape[0]


                if cfg.TRAIN.fov_scale_lr != '':
                    print(F_Xlr.shape)
                    pb = cfg.MODEL.patch_bank
                    wps = torch.sum(F_Xlr[:,:,xi,yi] * torch.tensor(pb).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(F_Xlr.device), dim=1).mean()
                    wpsn = (wps-pb[0])/(pb[-1]-pb[0])
                    print('wpsn: ', wpsn)
                    wpsn_mb += wpsn

                # split multi gpu collected dict into list to keep DataParall work for segmentation_module
                # print('patch_data_img_data_shape: ', patch_data['img_data'].shape)
                if mb_iter_count == 0:
                    patch_data_list = []
                    for idx in range(len(batch_data)):
                        patch_data_temp = dict()
                        patch_data_temp['img_data'] = torch.split(patch_data['img_data'], patch_data['img_data'].shape[0]//len(batch_data), dim=0)[idx]
                        patch_data_temp['seg_label'] = torch.split(patch_data['seg_label'], patch_data['seg_label'].shape[0]//len(batch_data), dim=0)[idx]
                        if cfg.MODEL.hard_fov_pred:
                            patch_data_temp['hard_max_idx'] = torch.split(patch_data['hard_max_idx'], patch_data['hard_max_idx'].shape[0]//len(batch_data), dim=0)[idx]
                        patch_data_list.append(patch_data_temp)
                else:
                    for idx in range(len(batch_data)):
                        patch_data_temp['img_data'] = torch.split(patch_data['img_data'], patch_data['img_data'].shape[0]//len(batch_data), dim=0)[idx]
                        patch_data_temp['seg_label'] = torch.split(patch_data['seg_label'], patch_data['seg_label'].shape[0]//len(batch_data), dim=0)[idx]
                        patch_data_list[idx]['img_data'] = torch.cat([patch_data_list[idx]['img_data'], patch_data_temp['img_data']])
                        patch_data_list[idx]['seg_label'] = torch.cat([patch_data_list[idx]['seg_label'], patch_data_temp['seg_label']])
                        if cfg.MODEL.hard_fov_pred:
                            patch_data_temp['hard_max_idx'] = torch.split(patch_data['hard_max_idx'], patch_data['hard_max_idx'].shape[0]//len(batch_data), dim=0)[idx]
                            patch_data_list[idx]['hard_max_idx'] = torch.cat([patch_data_list[idx]['hard_max_idx'], patch_data_temp['hard_max_idx']])
                    mb_iter_count += 1
                mb_iter_count = 0
                # forward pass
                # print('[patch_data_list_img_data_shape: ]', patch_data_list[0]['img_data'].shape)
                if single_gpu_mode:
                    loss, acc = segmentation_module(patch_data_list[0])
                else:
                    loss, acc = segmentation_module(patch_data_list)
                if cfg.MODEL.categorical:
                    # print('log_prob_act:', patch_data['log_prob_act'])
                    # print('ori loss:', loss)
                    if cfg.MODEL.inv_categorical:
                        loss = -patch_data['log_prob_act']*loss
                    else:
                        loss = patch_data['log_prob_act']*loss
                    # print('reinforced loss:', loss)
                if not single_gpu_mode:
                    loss = loss.mean()
                    acc = acc.mean()
                if cfg.TRAIN.entropy_regularisation:
                    loss += cfg.TRAIN.entropy_regularisation_weight*mbs_mean_entropy_reg
                if fov_location_batch_step//mbs == 1:
                    loss_step = loss.data
                    acc_step = acc.data
                else:
                    loss_step += loss.data
                    acc_step += acc.data


                if fov_location_batch_step == rand_location:
                    loss_retain = loss
                elif fov_location_batch_step != cfg.TRAIN.fov_location_step:
                    loss.backward()
                    if cfg.TRAIN.sync_location != 'rand':
                        for optimizer in optimizers:
                            optimizer.step()


                if fov_location_batch_step == cfg.TRAIN.fov_location_step:

                    if cfg.TRAIN.sync_location != 'none_sync':
                        # print('iter {}: bp at random retained location {}/{}, xi={}, yi={}'.format(i, rand_location, cfg.TRAIN.fov_location_step, xi, yi))
                        if cfg.TRAIN.sync_location == 'mean_mbs':
                            loss_retain.data = loss_step / (cfg.TRAIN.fov_location_step/mbs)
                        loss_retain.backward()
                    else:
                        loss.backward()
                    for optimizer in optimizers:
                        optimizer.step()
                    loss_step /= (cfg.TRAIN.fov_location_step/mbs)
                    acc_step /= (cfg.TRAIN.fov_location_step/mbs)
                    ave_total_loss.update(loss_step.data.item())
                    ave_acc.update(acc_step.data.item()*100)
                    fov_location_batch_step = 0
                    if not cfg.TRAIN.auto_fov_location_step and cfg.TRAIN.sync_location == 'rand':
                        rand_location = random.randint(2, cfg.TRAIN.fov_location_step-1)
                # print('iter {}: {}/{}/{} foveate points, xi={}, yi={}\n'.format(i, fov_location_batch_step, mb_idx, sudo_X_lr.shape[2]*sudo_X_lr.shape[3], xi, yi))

        else:
            # forward pass
            loss, acc = segmentation_module(batch_data)
            print()
            loss_step = loss.mean()
            acc_step = acc.mean()

            # Backward
            loss_step.backward()
            for optimizer in optimizers:
                optimizer.step()

            # update average loss and acc
            ave_total_loss.update(loss_step.data.item())
            ave_acc.update(acc_step.data.item()*100)

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()



        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            if cfg.MODEL.foveation:
                print('iter {}: bp at random retained location {}/{}, xi={}, yi={}'.format(i, rand_location, cfg.TRAIN.fov_location_step, xi, yi))
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

        fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
        if history is not None:
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(ave_total_loss.average())
            history['train']['acc'].append(ave_acc.average()/100)
            history['train']['print_grad'] = print_grad

        return ave_acc.average(), ave_total_loss.average()


def checkpoint(nets, cfg, epoch):
    print('Saving checkpoints...')

    if cfg.MODEL.foveation:
        (net_encoder, net_decoder, crit, net_foveater) = nets
        dict_foveater = net_foveater.state_dict()

        torch.save(
            dict_foveater,
            '{}/foveater_epoch_{}.pth'.format(cfg.DIR, epoch))

    else:
        (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))



def checkpoint_last(nets, cfg, epoch):
    print('Saving checkpoints...')

    if cfg.MODEL.foveation:
        (net_encoder, net_decoder, crit, net_foveater) = nets
        dict_foveater = net_foveater.state_dict()

        torch.save(
            dict_foveater,
            '{}/foveater_epoch_last.pth'.format(cfg.DIR))

    else:
        (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        dict_encoder,
        '{}/encoder_epoch_last.pth'.format(cfg.DIR))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_last.pth'.format(cfg.DIR))

def checkpoint_history(history, cfg, epoch):
    print('Saving history...')
    # save history as csv
    data_frame = pd.DataFrame(
        data={'train_loss': history['save']['epoch']
            , 'train_loss': history['save']['train_loss']
            , 'train_acc': history['save']['train_acc']
            , 'val_iou': history['save']['val_iou']
            , 'val_acc': history['save']['val_acc']
              }
    )
    data_frame.to_csv('{}/history_epoch_last.csv'.format(cfg.DIR),
                      index_label='epoch')

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    if cfg.MODEL.foveation:
        (net_encoder, net_decoder, crit, net_foveater) = nets
    else:
        (net_encoder, net_decoder, crit) = nets
    if cfg.TRAIN.optim.lower() == 'sgd':
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.SGD(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        if cfg.MODEL.foveation:
            optimizer_foveater = torch.optim.SGD(
                group_weight(net_foveater),
                lr=cfg.TRAIN.lr_foveater,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay_fov)
    elif cfg.TRAIN.optim.lower() == 'adam':
        optimizer_encoder = torch.optim.Adam(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.Adam(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            weight_decay=cfg.TRAIN.weight_decay)
        if cfg.MODEL.foveation:
            optimizer_foveater = torch.optim.Adam(
                group_weight(net_foveater),
                lr=cfg.TRAIN.lr_foveater,
                weight_decay=cfg.TRAIN.weight_decay_fov)

    if cfg.MODEL.foveation:
        return (optimizer_encoder, optimizer_decoder, optimizer_foveater)
    else:
        return (optimizer_encoder, optimizer_decoder)

def adjust_gms_tau(cur_iter, cfg, r=1e5):
    cfg.MODEL.gumbel_tau = max(0.1, float(np.exp(-1.*r*float(cur_iter))))
    print('adjusted_tau: ', cfg.MODEL.gumbel_tau)

def adjust_learning_rate(optimizers, cur_iter, cfg, lr_mbs = False, f_max_iter=1, lr_scale=1, wd_scale=1):
    # print('adjusted max_iter:', cfg.TRAIN.max_iters)
    scale_running_lr = ((1. - float(cur_iter) / f_max_iter) ** cfg.TRAIN.lr_pow)
    if not lr_mbs:
        scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    if cfg.TRAIN.fov_scale_lr != '':
        lr_scale = pow(lr_scale, cfg.TRAIN.fov_scale_pow)
        wd_scale = pow(wd_scale, cfg.TRAIN.fov_scale_pow)
        print('after fov_pow lr_scale={}, wd_scale={}'.format(lr_scale, wd_scale))
        print('original scale_running_lr: ', scale_running_lr)
        scale_running_lr *= lr_scale
        print('scaled scale_running_lr: ', scale_running_lr)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr
    if cfg.TRAIN.fov_scale_seg_only:
        scale_running_lr /= lr_scale
    cfg.TRAIN.running_lr_foveater = cfg.TRAIN.lr_foveater * scale_running_lr

    if cfg.MODEL.foveation:
        # cfg.TRAIN.running_lr_encoder /= cfg.TRAIN.fov_location_step
        # cfg.TRAIN.running_lr_decoder /= cfg.TRAIN.fov_location_step
        (optimizer_encoder, optimizer_decoder, optimizer_foveater) = optimizers
        for param_group in optimizer_foveater.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_foveater
            if cfg.TRAIN.fov_scale_weight_decay != '' and not cfg.TRAIN.fov_scale_seg_only:
                param_group['weight_decay'] = cfg.TRAIN.weight_decay_fov * wd_scale
    else:
        (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
        if cfg.TRAIN.fov_scale_weight_decay != '':
            param_group['weight_decay'] = cfg.TRAIN.weight_decay * wd_scale
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder
        if cfg.TRAIN.fov_scale_weight_decay != '':
            param_group['weight_decay'] = cfg.TRAIN.weight_decay * wd_scale


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
        dilate_rate=cfg.DATASET.segm_downsampling_rate)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)
    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
            in_channel=cfg.MODEL.in_dim,
            out_channel=len(cfg.MODEL.patch_bank),
            len_gpus=len(gpus),
            weights=cfg.MODEL.weights_foveater,
            cfg=cfg)

    # tensor
    writer = SummaryWriter('{}/tensorboard'.format(cfg.DIR))

    if cfg.DATASET.root_dataset == '/scratch0/chenjin/GLEASON2019_DATA/Data/':
        if cfg.TRAIN.loss_fun == 'DiceLoss':
            crit = DiceLoss()
        elif cfg.TRAIN.loss_fun == 'FocalLoss':
            crit = FocalLoss()
        elif cfg.TRAIN.loss_fun == 'DiceCoeff':
            crit = DiceCoeff()
        elif cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = OhemCrossEntropy(ignore_label=-1,
                                     thres=0.9,
                                     min_kept=100000,
                                     weight=None)
    elif 'ADE20K' in cfg.DATASET.root_dataset:
        crit = nn.NLLLoss(ignore_index=-2)
    elif 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                            1.0166, 0.9969, 0.9754, 1.0489,
                                            0.8786, 1.0023, 0.9539, 0.9843,
                                            1.1116, 0.9037, 1.0865, 1.0955,
                                            1.0865, 1.1529, 1.0507]).cuda()
            crit = OhemCrossEntropy(ignore_label=20,
                                         thres=0.9,
                                         min_kept=131072,
                                         weight=class_weights)
    elif 'DeepGlob' in cfg.DATASET.root_dataset and (cfg.TRAIN.loss_fun == 'FocalLoss' or cfg.TRAIN.loss_fun == 'OhemCrossEntropy'):
        if cfg.TRAIN.loss_fun == 'FocalLoss':
            crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
        elif cfg.TRAIN.loss_fun == 'OhemCrossEntropy':
            crit = OhemCrossEntropy(ignore_label=cfg.DATASET.ignore_index,
                                         thres=0.9,
                                         min_kept=131072)
    else:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            if cfg.DATASET.ignore_index != -2:
                crit = nn.NLLLoss(ignore_index=cfg.DATASET.ignore_index)
            else:
                crit = nn.NLLLoss(ignore_index=-2)
        else:
            if cfg.DATASET.ignore_index != -2:
                crit = nn.CrossEntropyLoss(ignore_index=cfg.DATASET.ignore_index)
            else:
                crit = nn.CrossEntropyLoss(ignore_index=-2)
    # crit = DiceCoeff()

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg, cfg.TRAIN.deep_sup_scale)
    elif cfg.MODEL.foveation:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg)

    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg, len_gpus=len(gpus))
        total_fov = sum([param.nelement() for param in foveation_module.parameters()])
        print('Number of FoveationModule params: %.2fM \n' % (total_fov / 1e6))

    total = sum([param.nelement() for param in segmentation_module.parameters()])
    print('Number of SegmentationModule params: %.2fM \n' % (total / 1e6))

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)


    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
        if cfg.MODEL.foveation:
            foveation_module = UserScatteredDataParallel(
                foveation_module,
                device_ids=gpus)
            patch_replication_callback(foveation_module)

    segmentation_module.cuda()
    if cfg.MODEL.foveation:
        foveation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    if cfg.MODEL.foveation:
        nets = (net_encoder, net_decoder, crit, net_foveater)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    if cfg.VAL.dice:
        history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'save': {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_iou': [], 'val_dice': [], 'val_acc': [], 'print_grad': None}}
    else:
        history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'save': {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_iou': [], 'val_dice': [], 'val_acc': [], 'print_grad': None}}

    if cfg.TRAIN.start_epoch > 0:
        history_previous_epoches = pd.read_csv('{}/history_epoch_{}.csv'.format(cfg.DIR, cfg.TRAIN.start_epoch))
        history['save']['epoch'] = list(history_previous_epoches['epoch'])
        history['save']['train_loss'] = list(history_previous_epoches['train_loss'])
        history['save']['train_acc'] = list(history_previous_epoches['train_acc'])
        history['save']['val_iou'] = list(history_previous_epoches['val_iou'])
        history['save']['val_acc'] = list(history_previous_epoches['val_acc'])
        # if cfg.VAL.dice:
        #     history['save']['val_dice'] = history_previous_epoches['val_dice']

    if not os.path.isdir(os.path.join(cfg.DIR, "Fov_probability_distribution")):
        os.makedirs(os.path.join(cfg.DIR, "Fov_probability_distribution"))
    f_prob = []
    for p in range(len(cfg.MODEL.patch_bank)):
        f = open(os.path.join(cfg.DIR, 'Fov_probability_distribution', 'patch_{}_distribution.txt'.format(p)), 'w')
        f.close()

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        if cfg.MODEL.foveation:
            train_acc, train_loss = train(segmentation_module, iterator_train, optimizers, epoch+1, cfg, history, foveation_module)
            if history['train']['print_grad'] is not None and type(history['train']['print_grad']) is not torch.Tensor:
                if history['train']['print_grad']['layer1_grad'] is not None and history['train']['print_grad']['layer1_grad'][history['train']['print_grad']['layer1_grad']>0].numel() > 0:
                    writer.add_histogram('Print non-zero gradient (layer1) histogram', history['train']['print_grad']['layer1_grad'][history['train']['print_grad']['layer1_grad']>0], epoch+1)
                    writer.add_histogram('Print gradient (layer1) histogram', history['train']['print_grad']['layer1_grad'], epoch+1)
                    writer.add_scalar('Percentage none-zero gradients (layer1)', history['train']['print_grad']['layer1_grad'][history['train']['print_grad']['layer1_grad']>0].numel()/history['train']['print_grad']['layer1_grad'].numel(), epoch+1)
                    writer.add_image('Print_grad_Fov_softmax_layer1(normalized_b0_p0)', (history['train']['print_grad']['layer1_grad'][0][0]-history['train']['print_grad']['layer1_grad'][0][0].min())/(history['train']['print_grad']['layer1_grad'][0][0].max()-history['train']['print_grad']['layer1_grad'][0][0].min()), epoch+1, dataformats='HW')
                if history['train']['print_grad']['layer2_grad'] is not None and history['train']['print_grad']['layer2_grad'][history['train']['print_grad']['layer2_grad']>0].numel() > 0:
                    writer.add_histogram('Print non-zero gradient (layer2) histogram', history['train']['print_grad']['layer2_grad'][history['train']['print_grad']['layer2_grad']>0], epoch+1)
                    writer.add_histogram('Print gradient (layer2) histogram', history['train']['print_grad']['layer2_grad'], epoch+1)
                    writer.add_scalar('Percentage none-zero gradients (layer2)', history['train']['print_grad']['layer2_grad'][history['train']['print_grad']['layer2_grad']>0].numel()/history['train']['print_grad']['layer2_grad'].numel(), epoch+1)
                    writer.add_image('Print_grad_Fov_softmax_layer2(normalized_b0_p0)', (history['train']['print_grad']['layer2_grad'][0][0]-history['train']['print_grad']['layer2_grad'][0][0].min())/(history['train']['print_grad']['layer2_grad'][0][0].max()-history['train']['print_grad']['layer2_grad'][0][0].min()), epoch+1, dataformats='HW')
                if history['train']['print_grad']['layer3_grad'] is not None and history['train']['print_grad']['layer3_grad'][history['train']['print_grad']['layer3_grad']>0].numel() > 0:
                    writer.add_histogram('Print non-zero gradient (layer3) histogram', history['train']['print_grad']['layer3_grad'][history['train']['print_grad']['layer3_grad']>0], epoch+1)
                    writer.add_histogram('Print gradient (layer3) histogram', history['train']['print_grad']['layer3_grad'], epoch+1)
                    writer.add_scalar('Percentage none-zero gradients (layer3)', history['train']['print_grad']['layer3_grad'][history['train']['print_grad']['layer3_grad']>0].numel()/history['train']['print_grad']['layer3_grad'].numel(), epoch+1)
                    writer.add_image('Print_grad_Fov_softmax_layer3(normalized_b0_p0)', (history['train']['print_grad']['layer3_grad'][0][0]-history['train']['print_grad']['layer3_grad'][0][0].min())/(history['train']['print_grad']['layer3_grad'][0][0].max()-history['train']['print_grad']['layer3_grad'][0][0].min()), epoch+1, dataformats='HW')

        else:
            train_acc, train_loss = train(segmentation_module, iterator_train, optimizers, epoch+1, cfg, history)
        # checkpointing

        if (epoch+1) % cfg.TRAIN.checkpoint_per_epoch == 0:
            checkpoint(nets, cfg, epoch+1)
            checkpoint_last(nets, cfg, epoch+1)
        else:
            checkpoint_last(nets, cfg, epoch+1)


        if (epoch+1) % cfg.TRAIN.eval_per_epoch == 0:
            # eval during train
            if cfg.VAL.multipro:
                if cfg.MODEL.foveation:
                    if cfg.VAL.all_F_Xlr_time:
                        val_iou, val_acc, F_Xlr_all, F_Xlr_score_flat_all = eval_during_train_multipro(cfg, gpus)
                    else:
                        val_iou, val_acc, F_Xlr, F_Xlr_score_flat = eval_during_train_multipro(cfg, gpus)
                else:
                    val_iou, val_acc = eval_during_train_multipro(cfg, gpus)
            else:
                if cfg.VAL.dice:
                    if cfg.MODEL.foveation:
                        if cfg.VAL.all_F_Xlr_time:
                            val_iou, val_dice, val_acc, F_Xlr_all, F_Xlr_score_flat_all = eval_during_train(cfg)
                        else:
                            val_iou, val_dice, val_acc, F_Xlr, F_Xlr_score_flat = eval_during_train(cfg)
                    else:
                        val_iou, val_dice, val_acc = eval_during_train(cfg)
                else:
                    if cfg.MODEL.foveation:
                        if cfg.VAL.all_F_Xlr_time:
                            val_iou, val_acc, F_Xlr_all, F_Xlr_score_flat_all = eval_during_train(cfg)
                        else:
                            val_iou, val_acc, F_Xlr, F_Xlr_score_flat = eval_during_train(cfg)
                    else:
                        val_iou, val_acc = eval_during_train(cfg)

            history['save']['epoch'].append(epoch+1)
            history['save']['train_loss'].append(history['train']['loss'][-1])
            history['save']['train_acc'].append(history['train']['acc'][-1]*100)
            history['save']['val_iou'].append(val_iou)
            if cfg.VAL.dice:
                history['save']['val_dice'].append(val_dice)
            history['save']['val_acc'].append(val_acc)
            # write to tensorboard
            writer.add_scalar('Loss/train', history['train']['loss'][-1], epoch+1)
            writer.add_scalar('Acc/train', history['train']['acc'][-1]*100, epoch+1)
            writer.add_scalar('Acc/val', val_acc, epoch+1)
            writer.add_scalar('mIoU/val', val_iou, epoch+1)
            if cfg.VAL.dice:
                writer.add_scalar('mDice/val', val_acc, epoch+1)
            if cfg.VAL.all_F_Xlr_time:
                print('=============F_Xlr_score_flat_all================\n', F_Xlr_score_flat_all.shape)
                for p in range(F_Xlr_score_flat_all.shape[0]):
                    # add small artifact to modify range, because no range flag in add_histogram
                    F_Xlr_score_flat_all[p][0] = 0
                    F_Xlr_score_flat_all[p][-1] = 1
                    writer.add_histogram('Patch_{} probability histogram'.format(p), F_Xlr_score_flat_all[p], epoch+1)
                    f = open(os.path.join(cfg.DIR, 'Fov_probability_distribution', 'patch_{}_distribution.txt'.format(p)), 'a')
                    if epoch == 0:
                        f.write('epoch/ bins: {}\n'.format(np.histogram(F_Xlr_score_flat_all[p], bins=10, range=(0, 1))[1]))
                    f.write('epoch {}: {}\n'.format(epoch+1, np.histogram(F_Xlr_score_flat_all[p], bins=10, range=(0, 1))[0]/sum(np.histogram(F_Xlr_score_flat_all[p], bins=10, range=(0, 1))[0])))
                    f.close()
                writer.add_histogram('Patch_All probability histogram', F_Xlr_score_flat_all, epoch+1)
            else:
                for p in range(F_Xlr_score_flat_all.shape[0]):
                    F_Xlr_score_flat[p][0] = 0
                    F_Xlr_score_flat[p][-1] = 1
                    writer.add_histogram('Patch_{} probability histogram'.format(p), F_Xlr_score_flat[p], epoch+1)
                writer.add_histogram('Patch_All probability histogram', F_Xlr_score_flat, epoch+1)
        else:
            history['save']['epoch'].append(epoch+1)
            history['save']['train_loss'].append(history['train']['loss'][-1])
            history['save']['train_acc'].append(history['train']['acc'][-1]*100)
            history['save']['val_iou'].append('')
            if cfg.VAL.dice:
                history['save']['val_dice'].append('')
            history['save']['val_acc'].append('')
            # write to tensorboard
            writer.add_scalar('Loss/train', history['train']['loss'][-1], epoch+1)
            writer.add_scalar('Acc/train', history['train']['acc'][-1]*100, epoch+1)
            # writer.add_scalar('Acc/val', '', epoch+1)
            # writer.add_scalar('mIoU/val', '', epoch+1)

        # saving history
        checkpoint_history(history, cfg, epoch+1)

        if (epoch+1) % cfg.TRAIN.eval_per_epoch == 0:
            # output F_Xlr
            if cfg.MODEL.foveation:
                # save time series F_Xlr (t,b,d,w,h)
                if epoch == 0 or epoch == cfg.TRAIN.start_epoch:
                    if cfg.VAL.all_F_Xlr_time:
                        F_Xlr_time_all = []
                        for val_idx in range(len(F_Xlr_all)):
                            F_Xlr_time_all.append(F_Xlr_all[val_idx][0])
                    else:
                        F_Xlr_time = F_Xlr
                else:
                    if cfg.VAL.all_F_Xlr_time:
                        for val_idx in range(len(F_Xlr_all)):
                            F_Xlr_time_all[val_idx] = np.concatenate((F_Xlr_time_all[val_idx], F_Xlr_all[val_idx][0]), axis=0)
                    else:
                        F_Xlr_time = np.concatenate((F_Xlr_time, F_Xlr), axis=0)
                if cfg.VAL.all_F_Xlr_time:
                    for val_idx in range(len(F_Xlr_all)):
                        print('F_Xlr_time_{}'.format(F_Xlr_all[val_idx][1]), F_Xlr_time_all[val_idx].shape)
                        if not os.path.isdir(os.path.join(cfg.DIR, "F_Xlr_time_all_vals")):
                            os.makedirs(os.path.join(cfg.DIR, "F_Xlr_time_all_vals"))
                        np.save('{}/F_Xlr_time_all_vals/F_Xlr_time_last_{}.npy'.format(cfg.DIR, F_Xlr_all[val_idx][1]), F_Xlr_time_all[val_idx])
                else:
                    print('F_Xlr_time', F_Xlr_time.shape)
                    np.save('{}/F_Xlr_time_last.npy'.format(cfg.DIR), F_Xlr_time)

    if not cfg.TRAIN.save_checkpoint:
        os.remove('{}/encoder_epoch_last.pth'.format(cfg.DIR))
        os.remove('{}/decoder_epoch_last.pth'.format(cfg.DIR))
    print('Training Done!')
    writer.close()

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/foveation-cityscape-hrnetv2.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    if cfg.TRAIN.auto_batch == 'auto10':
        # asign 10G per gpu estimated by: can take about 10e6 pixels with hrnetv2
        cfg.TRAIN.batch_size_per_gpu = int((1e6*0.65) // (cfg.DATASET.imgSizes[0]*cfg.DATASET.imgSizes[0]))
        gpus = parse_devices(args.gpus)
        num_gpu = len(gpus)
        num_data = len([x for x in open(cfg.DATASET.list_train, 'r')])
        print(num_data, num_gpu, cfg.TRAIN.batch_size_per_gpu)
        cfg.TRAIN.epoch_iters = int(num_data // (num_gpu*cfg.TRAIN.batch_size_per_gpu))

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
        if cfg.MODEL.foveation:
            cfg.MODEL.weights_foveater = os.path.join(
                cfg.DIR, 'foveater_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
