# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.io import loadmat
# Our libs
from config import cfg
from dataset import ValDataset, imresize, b_imresize, patch_loader
from models import ModelBuilder, SegmentationModule, FovSegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from criterion import OhemCrossEntropy, DiceCoeff, DiceLoss, FocalLoss
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))

def visualize_result_fov(data, foveated_expection, dir_result):
    (img, F_Xlr, info) = data

    # segmentation
    F_Xlr_color = colorEncode(F_Xlr, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, F_Xlr_color, foveated_expection),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))

def evaluate(segmentation_module, loader, cfg, gpu, foveation_module=None):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    patch_bank = list((float(cfg.VAL.expand_prediection_rate_patch)*np.array(cfg.MODEL.patch_bank)).astype(int))
    patch_bank = async_copy_to(patch_bank, gpu)
    # print('eval_patch_bank_1:', patch_bank)

    segmentation_module.eval()
    if cfg.MODEL.foveation:
        foveation_module.eval()

    if cfg.VAL.all_F_Xlr_time:
        F_Xlr_all = []
        F_Xlr_score_flat_all = None

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # for ib in range(len(batch_data)):
            # print('img_data shape: ',  batch_data[ib]['img_data'][0].shape)
            # print('seg_label shape: ',  batch_data[ib]['seg_label'][0].shape)
        # process data
        # NOTE: here different to training, only batch_data[0] is assigned, i.e. not it's only able to process batch_size = 1 in inference
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        img_resized_list_unnorm = batch_data['img_data_unnorm']
        # note for foveation resize not applied, i.e. both seg_label and img_data are at original size
        if cfg.VAL.visualize and cfg.MODEL.foveation:
            foveated_expection = torch.zeros(batch_data['img_ori'].shape)
            if cfg.VAL.hard_max_fov:
                foveated_expection_temp = torch.cat([foveated_expection.unsqueeze(0), foveated_expection.unsqueeze(0)])
                foveated_expection_weight =  torch.zeros(foveated_expection_temp.shape[0:-1]) # 2,w,h
            else:
                overlap_count = torch.zeros(batch_data['img_ori'].shape)

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores_tmp = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)
            scores_tmp = async_copy_to(scores_tmp, gpu)

            if cfg.VAL.max_score:
                scores_tmp_2 = torch.cat([scores_tmp.unsqueeze(0), scores_tmp.unsqueeze(0)])
                scores_tmp_2 = async_copy_to(scores_tmp_2, gpu)

            if cfg.VAL.approx_pred_Fxlr_by_ensemble or cfg.VAL.F_Xlr_low_scale != 0:
                fov_map_scale_temp = cfg.MODEL.fov_map_scale
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    scores_ensemble = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
                    scores_ensemble = async_copy_to(scores_ensemble, gpu)
                    approx_pred_Fxlr_iter = len(patch_bank)
                # create fake feed_dict
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img_resized_list[0]
                feed_dict['img_data_unnorm'] = img_resized_list_unnorm[0]
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                # get F_Xlr at original high resolution fov_map_scale # b,d,w,h
                X = feed_dict['img_data'] # NOTE only support test image = 1
                fov_map_scale = cfg.MODEL.fov_map_scale
                X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')
                feed_dict['cor_info'] = (tuple([0]), tuple([0]))
                patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                F_Xlr_ori = F_Xlr.clone()
                # print(F_Xlr.size())
                # scale F_Xlr to size of score b,d,W,H
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    F_Xlr_scale = b_imresize(F_Xlr, (segSize[0], segSize[1]), interp='nearest')
                if cfg.VAL.F_Xlr_low_scale != 0:
                    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!Fist detect F_Xlr_low_scale')
                    F_Xlr_low_res = b_imresize(F_Xlr, (round(X.shape[2]/cfg.VAL.F_Xlr_low_scale), round(X.shape[3]/(cfg.VAL.F_Xlr_low_scale*cfg.MODEL.patch_ap))), interp='bilinear')
                    cfg.MODEL.fov_map_scale = cfg.VAL.F_Xlr_low_scale
                    approx_pred_Fxlr_iter = 1
                    # print('cfg.VAL.F_Xlr_low_scale:', cfg.VAL.F_Xlr_low_scale)
            else:
                approx_pred_Fxlr_iter = 1

            for pred_iter in range(approx_pred_Fxlr_iter):
                if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                    cfg.MODEL.fov_map_scale = patch_bank[0]
                    cfg.MODEL.one_hot_patch = [0]*len(patch_bank)
                    cfg.MODEL.one_hot_patch[pred_iter] = 1
                for idx in range(len(img_resized_list)):
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img_resized_list[idx]
                    feed_dict['img_data_unnorm'] = img_resized_list_unnorm[idx]
                    if cfg.VAL.F_Xlr_low_scale != 0:
                        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!ADD')
                        feed_dict['F_Xlr_low_res'] = F_Xlr_low_res
                        # print('F_Xlr_low_res_size:', feed_dict['F_Xlr_low_res'].size())
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, gpu)

                    # Foveation
                    if cfg.MODEL.foveation:
                        X, Y = feed_dict['img_data'], feed_dict['seg_label']
                        X_unnorm = feed_dict['img_data_unnorm']
                        with torch.no_grad():
                            # print('eval_patch_bank_2:', patch_bank)
                            patch_segSize = (patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                            # print('eval_patch_segSize: ', patch_segSize)
                            patch_scores = torch.zeros(1, cfg.DATASET.num_class, patch_segSize[0], patch_segSize[1])
                            patch_scores = async_copy_to(patch_scores, gpu)

                        fov_map_scale = cfg.MODEL.fov_map_scale
                        # NOTE: although here we use batch imresize yet in practical batch size for X = 1
                        X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*cfg.MODEL.patch_ap))), interp='bilinear')

                        # foveation (crop as you go)
                        if cfg.VAL.F_Xlr_only:
                            feed_dict['cor_info'] = (tuple([0]), tuple([0]))
                            patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                        else:
                            # mb_count = 0
                            # xi_mb = []
                            # yi_mb = []
                            # mbs = cfg.TRAIN.mini_batch_size
                            for xi in range(X_lr.shape[2]):
                                for yi in range(X_lr.shape[3]):
                                    # print('current location: xi={}, yi={}'.format(xi, yi))
                                    # if mb_count < mbs-1:
                                    #     xi_mb.append(xi)
                                    #     yi_mb.append(yi)
                                    #     mb_count += 1
                                    #     continue
                                    # else:
                                    #     xi_mb.append(xi)
                                    #     yi_mb.append(yi)
                                    #     xi_mb_in = tuple(xi_mb)
                                    #     yi_mb_in = tuple(yi_mb)
                                    #     mb_count = 0
                                    #     xi_mb = []
                                    #     yi_mb = []
                                    # feed_dict['cor_info'] = (xi_mb_in, yi_mb_in)
                                    feed_dict['cor_info'] = (tuple([xi]), tuple([yi]))
                                    if cfg.VAL.visualize:
                                        patch_data, F_Xlr, Y_patch_cord, X_patches_cords, X_patches_unnorm = foveation_module(feed_dict, train_mode=False)
                                    else:
                                        patch_data, F_Xlr, Y_patch_cord = foveation_module(feed_dict, train_mode=False)
                                    # print('patch_data_shape: ', patch_data['img_data'].shape)
                                    # TODO: foveation (pre_cropped available)
                                    patch_scores = segmentation_module(patch_data, segSize=patch_segSize)
                                    # print('patch_scores_shape: ', patch_scores.shape)

                                    cx_Y, cy_Y, patch_size_Y, p_y_w, p_y_h = Y_patch_cord
                                    if cfg.MODEL.fov_padding:
                                        # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                        scores_tmp_pad = torch.zeros(scores_tmp.shape)
                                        scores_tmp_pad = F.pad(scores_tmp_pad, (p_y_w,p_y_w,p_y_h,p_y_h))
                                        scores_tmp_pad = async_copy_to(scores_tmp_pad, gpu)
                                        # print('scores_tmp_pad shape: ', scores_tmp_pad.shape)
                                    patch_size_Y_x = patch_size_Y
                                    patch_size_Y_y = patch_size_Y*cfg.MODEL.patch_ap
                                    if not cfg.VAL.max_score:
                                        if cfg.MODEL.fov_padding:
                                            scores_tmp_pad = scores_tmp_pad*0
                                            scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores.clone()
                                            scores_tmp = torch.add(scores_tmp, scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w])
                                        else:
                                            scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = torch.add(scores_tmp[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y], patch_scores)
                                    else:
                                        if cfg.MODEL.fov_padding:
                                            scores_tmp_pad = scores_tmp_pad*0
                                            scores_tmp_pad[:, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                            scores_tmp_2[1] = scores_tmp_pad[:, :, p_y_h:scores_tmp_pad.shape[2]-p_y_h, p_y_w:scores_tmp_pad.shape[3]-p_y_w]
                                        else:
                                            scores_tmp_2[1, :, :, cx_Y:cx_Y+patch_size_Y_x, cy_Y:cy_Y+patch_size_Y_y] = patch_scores
                                        max_class_scores_tmp_2_0, _ = torch.max(scores_tmp_2[0], dim=1)
                                        max_class_scores_tmp_2_1, _ = torch.max(scores_tmp_2[1], dim=1)
                                        # 2,B,W,H, B=1
                                        max_class_scores_tmp_2 = torch.cat([max_class_scores_tmp_2_0.unsqueeze(0), max_class_scores_tmp_2_1.unsqueeze(0)])
                                        # get patch idx of max(max(score))
                                        # patch_idx_by_score.shape = B,W,H; B=1
                                        _, patch_idx_by_score = torch.max(max_class_scores_tmp_2, dim=0)
                                        scores_tmp_2_patch_idx = patch_idx_by_score.unsqueeze(1).unsqueeze(0).expand(scores_tmp_2.shape)
                                        scores_tmp_2[0] = scores_tmp_2.gather(0, scores_tmp_2_patch_idx)[0]
                                        scores_tmp_2[1] = torch.zeros(scores_tmp_2[0].shape)

                                    if cfg.VAL.visualize:
                                        if cfg.VAL.central_crop:
                                            cx_0, cy_0, patch_size_0, p_y_w, p_y_h = Y_patch_cord
                                        if cfg.VAL.hard_max_fov:
                                            weight_s, max_s = torch.max(F_Xlr[0,:,xi,yi], dim=0)
                                            if cfg.MODEL.hard_fov or cfg.MODEL.categorical:
                                                max_s = 0
                                            cx, cy, patch_size, p_w, p_h = X_patches_cords[max_s]
                                            X_patch = b_imresize(X_patches_unnorm[:,max_s,:,:,:], (patch_size, patch_size), interp='nearest')
                                            X_patch = X_patch[0]
                                            print('X_patch_shape: ', X_patch.shape)
                                            # c,w,h
                                            weighed_patch = X_patch.permute(1,2,0).cpu()
                                            # w,h
                                            patch_weight = weight_s.unsqueeze(-1).expand(*weighed_patch.shape[0:-1])
                                        else: # soft fov - max_score=False mode not currently supported
                                            cx_w, cy_w, patch_size_w = 0, 0, 0
                                            for i in range(len(X_patches_cords)):
                                                cx, cy, patch_size, p_w, p_h = X_patches_cords[i]
                                                w = F_Xlr[0,i,xi,yi]
                                                cx_w += w*cx
                                                cy_w += w*cy
                                                patch_size_w += w*patch_size
                                            cx, cy, patch_size = int(cx_w), int(cy_w), int(patch_size_w)
                                            # patch_size = int(torch.sum(F_Xlr[0,:,xi,yi] * torch.FloatTensor(cfg.MODEL.patch_bank)))
                                            if cfg.MODEL.fov_padding:
                                                fov_map_scale = cfg.MODEL.fov_map_scale
                                                # p = patch_size
                                                cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                                cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                                X_unnorm_pad = F.pad(X_unnorm, (p_w,p_w,p_h,p_h))
                                                crop_patch = X_unnorm_pad[:, :, cx_p:cx_p+patch_size, cy_p:cy_p+patch_size]
                                            else:
                                                crop_patch = X_unnorm[:, :, cx:cx+patch_size, cy:cy+patch_size]
                                            X_patch = b_imresize(crop_patch, (patch_size_0,patch_size_0), interp='bilinear')
                                            X_patch = b_imresize(X_patch, (patch_size, patch_size), interp='nearest')
                                            X_patch = X_patch[0]
                                            print('X_patch_shape: ', X_patch.shape)
                                            # c,w,h
                                            weighed_patch = X_patch.permute(1,2,0).cpu()

                                        if cfg.VAL.foveated_expection:
                                            if cfg.MODEL.fov_padding:
                                                fov_map_scale = cfg.MODEL.fov_map_scale
                                                # p = patch_size
                                                cx_p = xi*fov_map_scale + patch_size_Y//2 - patch_size//2 + p_h
                                                cy_p = yi*(fov_map_scale*cfg.MODEL.patch_ap) + patch_size_Y//2 - patch_size//2 + p_w
                                                # C,W,H
                                                foveated_expection_temp_pad = torch.zeros(foveated_expection_temp.shape[3],foveated_expection_temp.shape[1],foveated_expection_temp.shape[2])
                                                foveated_expection_temp_pad = F.pad(foveated_expection_temp_pad, (p_w,p_w,p_h,p_h))
                                                # print('foveated_expection_temp_pad:', foveated_expection_temp_pad.shape)
                                                # print('cx_p, cy_p, patch_size:', cx_p, cy_p, patch_size)
                                                # W,H,C
                                                foveated_expection_temp_pad = foveated_expection_temp_pad.permute(1,2,0)
                                                foveated_expection_temp_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size, :] = weighed_patch
                                                foveated_expection_temp[1] = foveated_expection_temp_pad[p_h:-p_h, p_w:-p_w, :]
                                                if cfg.VAL.central_crop:
                                                    # p_y = max(patch_bank[0], patch_bank[0]*cfg.MODEL.patch_ap)
                                                    foveated_expection_temp_pad_y = foveated_expection_temp[1].clone() # W,H,C
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(2,0,1) # C,W,H
                                                    foveated_expection_temp_pad_y = F.pad(foveated_expection_temp_pad_y, (p_y_w,p_y_w,p_y_h,p_y_h))
                                                    foveated_expection_temp_temp = foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0].clone()
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y*0
                                                    foveated_expection_temp_pad_y[:, cx_0:cx_0+patch_size_0, cy_0:cy_0+patch_size_0] = foveated_expection_temp_temp
                                                    foveated_expection_temp_pad_y = foveated_expection_temp_pad_y.permute(1,2,0) # W,H,C
                                                    foveated_expection_temp[1] = foveated_expection_temp_pad_y[p_y_h:foveated_expection_temp_pad_y.shape[0]-p_y_h, p_y_w:foveated_expection_temp_pad_y.shape[1]-p_y_w, :]
                                                    print('max: ', torch.max(foveated_expection_temp_temp))
                                                    print('min: ', torch.min(foveated_expection_temp_temp))
                                                    # if torch.min(foveated_expection_temp_temp) == 0:
                                                    #     print(foveated_expection_temp_temp)
                                                    #     raise Exception('weighted patch may wrong')
                                                if cfg.VAL.hard_max_fov:
                                                    # W,H
                                                    foveated_expection_weight_pad = torch.zeros(foveated_expection_temp_pad.shape[0:-1])
                                                    foveated_expection_weight_pad[cx_p:cx_p+patch_size, cy_p:cy_p+patch_size] = patch_weight
                                                    foveated_expection_weight[1] = foveated_expection_weight_pad[p_h:-p_h, p_w:-p_w]
                                            else:
                                                # W,H,C
                                                foveated_expection_temp[1, cx:cx+patch_size, cy:cy+patch_size, :] = weighed_patch
                                                if cfg.VAL.hard_max_fov:
                                                    # W,H
                                                    foveated_expection_weight[1, cx:cx+patch_size, cy:cy+patch_size] = patch_weight
                                            if cfg.VAL.hard_max_fov:
                                                foveated_expection_weight[0], max_w_idx = torch.max(foveated_expection_weight, dim=0)
                                            if not cfg.VAL.max_score:
                                                max_w_idx = max_w_idx.unsqueeze(0).unsqueeze(-1).expand(*foveated_expection_temp.shape)
                                                # max_w_idx_w = max_w_idx.unsqueeze(0).expand(*foveated_expection_weight.shape)
                                                foveated_expection = foveated_expection_temp.gather(0, max_w_idx)[0]
                                            else:
                                                max_s_idx = patch_idx_by_score.unsqueeze(-1).expand(*foveated_expection_temp.shape).cpu()
                                                foveated_expection = foveated_expection_temp.gather(0, max_s_idx)[0]

                                            # foveated_expection_weight[0] = foveated_expection_weight.gather(0, max_w_idx_w).squeeze(0)
                                            foveated_expection_temp[0] = foveated_expection
                                            foveated_expection_temp[1] = torch.zeros(foveated_expection_temp[0].shape)
                                            if cfg.VAL.hard_max_fov:
                                                foveated_expection_weight[1] = torch.zeros(foveated_expection_weight[0].shape)


                                        # else:
                                        #     for s in range(len(X_patches_cords)):
                                        #         cx, cy, patch_size, p = X_patches_cords[s]
                                        #         # X_patches_unnorm: b,d,c,w,h
                                        #         X_patch = b_imresize(X_patches_unnorm[:,s,:,:,:], (patch_size, patch_size), interp='nearest')
                                        #         # X_patch: b,c,w,h
                                        #         # NOTE: current version only appliable for batch size = 1
                                        #         X_patch = X_patch[0]
                                        #         # c,w,h
                                        #         # TODO: check is this right??? should it be F_Xlr[:,s,xi,yi] NOT 1-F_Xlr[:,s,xi,yi] ?
                                        #         weighed_patch = (1-F_Xlr[:,s,xi,yi]).unsqueeze(-1).unsqueeze(-1).expand(*X_patch.size())*X_patch
                                        #         # w,h,c
                                        #         weighed_patch = weighed_patch.permute(1,2,0).cpu()
                                        #         foveated_expection[cx:cx+patch_size, cy:cy+patch_size, :] = foveated_expection[cx:cx+patch_size, cy:cy+patch_size, :] + weighed_patch
                                        #         overlap_count[cx:cx+patch_size, cy:cy+patch_size, :] += torch.ones_like(weighed_patch)

                                        # print('{}/{} foveate points, xi={}, yi={}\n'.format(xi*X_lr.shape[3]+yi, X_lr.shape[2]*X_lr.shape[3], xi, yi))
                            if cfg.VAL.max_score:
                                scores_tmp = scores_tmp_2[0]
                        # print('F_Xlr: ', F_Xlr.shape)
                        # print(F_Xlr)
                    # non foveation mode
                    else:
                        # forward pass
                        scores_tmp = segmentation_module(feed_dict, segSize=segSize)

                    scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                    if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                        scores_ensemble = scores_ensemble + scores * F_Xlr_scale[:,pred_iter,:,:]

            if cfg.VAL.approx_pred_Fxlr_by_ensemble:
                scores = scores_ensemble
                cfg.MODEL.fov_map_scale = fov_map_scale_temp
            if cfg.VAL.F_Xlr_low_scale != 0:
                cfg.MODEL.fov_map_scale = fov_map_scale_temp
                F_Xlr = F_Xlr_ori
            if cfg.VAL.ensemble:
                if not os.path.isdir(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores')):
                    os.makedirs(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores'))
                np.save(os.path.join(cfg.DIR, "{}result_{}".format(cfg.VAL.rename_eval_folder, cfg.VAL.checkpoint), 'scores', batch_data['info'].split('/')[-1]), scores.cpu())
            _, pred = torch.max(scores, dim=1)
            # w,h
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)


        acc, pix = accuracy(pred, seg_label)
        if 'CITYSCAPES' in cfg.DATASET.root_dataset:
            intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=20-1)
        else:
            if cfg.DATASET.ignore_index != -2:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class, ignore_index=cfg.DATASET.ignore_index)
            else:
                intersection, union, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        if cfg.MODEL.foveation:
            if cfg.MODEL.gumbel_softmax:
                F_Xlr = F_Xlr.exp()

            # d,w,h
            F_Xlr_score = F_Xlr.clone()
            # d, w*h
            # print('F_Xlr_score_shape',F_Xlr_score.shape)
            F_Xlr_score_flat = F_Xlr_score.reshape(F_Xlr.size(1),-1)
            # print('F_Xlr_score_flat_shape',F_Xlr_score_flat.shape)
            patch_bank_F_Xlr = torch.tensor(cfg.MODEL.patch_bank).to(F_Xlr.device)
            F_Xlr = patch_bank_F_Xlr.unsqueeze(-1).unsqueeze(-1).float()*(F_Xlr.squeeze(0)).float()
            F_Xlr = as_numpy(F_Xlr.cpu())
            # F_Xlr = as_numpy(F_Xlr.squeeze(0).cpu())
            # print('F_Xlr_np', F_Xlr.shape)
            # d,w,h
            F_Xlr = np.sum(F_Xlr,axis=0)
            # w,h
            # print('F_Xlr_sum', F_Xlr.shape)
            # print(F_Xlr)
            F_Xlr = np.expand_dims(F_Xlr,axis=0)
            # print('F_Xlr_expand_dims', F_Xlr.shape)

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )
            if cfg.MODEL.foveation:
                foveated_expection = foveated_expection / overlap_count
                visualize_result_fov(
                    (batch_data['img_ori'], b_imresize(1-F_Xlr, (segSize[0], segSize[1]), interp='nearest'), batch_data['info']),
                    foveated_expection,
                    os.path.join(cfg.DIR, 'result_foveation')
                )

        if cfg.VAL.all_F_Xlr_time:
            F_Xlr_all.append((F_Xlr, batch_data['info'].split('/')[-1].split('.')[0]))
            if F_Xlr_score_flat_all is None:
                F_Xlr_score_flat_all = F_Xlr_score_flat
            else:
                F_Xlr_score_flat_all = torch.cat([F_Xlr_score_flat_all,F_Xlr_score_flat],axis=1)
        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    if cfg.VAL.dice:
        dice = (2 * intersection_meter.sum) / (union_meter.sum + intersection_meter.sum + 1e-10)
    # for i, _iou in enumerate(iou):
    #     print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    if cfg.VAL.dice:
        print('Mean IoU: {:.4f}, Mean Dice: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), dice.mean(), acc_meter.average()*100, time_meter.average()))
    else:
        print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))

    # implemented for eval during trainig
    if cfg.VAL.dice:
        if cfg.MODEL.foveation:
            if cfg.VAL.all_F_Xlr_time:
                return iou.mean(), dice.mean(), acc_meter.average()*100, F_Xlr_all, np.array(as_numpy(F_Xlr_score_flat_all))
            else:
                return iou.mean(), dice.mean(), acc_meter.average()*100, F_Xlr, F_Xlr_score_flat
        else:
            return iou.mean(), dice.mean(), acc_meter.average()*100
    else:
        if cfg.MODEL.foveation:
            if cfg.VAL.all_F_Xlr_time:
                return iou.mean(), acc_meter.average()*100, F_Xlr_all, np.array(as_numpy(F_Xlr_score_flat_all))
            else:
                return iou.mean(), acc_meter.average()*100, F_Xlr, F_Xlr_score_flat
        else:
            return iou.mean(), acc_meter.average()*100

def eval_during_train(cfg, gpu=0):
    torch.cuda.set_device(gpu)
    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    # load foveation weights
    if cfg.MODEL.foveation:
        weights=cfg.MODEL.weights_foveater = os.path.join(
            cfg.DIR, 'foveater_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_foveater), "checkpoint does not exitst!"
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"


    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
        in_channel=cfg.MODEL.in_dim,
        out_channel=len(cfg.MODEL.patch_bank),
        weights=cfg.MODEL.weights_foveater,
        cfg=cfg)

    if 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=19)
    elif 'Digest' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=-2)
    elif cfg.TRAIN.loss_fun == 'FocalLoss' and 'DeepGlob' in cfg.DATASET.root_dataset:
        crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
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

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)
    if cfg.MODEL.foveation:
        foveation_module = FovSegmentationModule(net_foveater, cfg)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()
    if cfg.MODEL.foveation:
        foveation_module.cuda()

    # Main loop
    if cfg.MODEL.foveation:
        if cfg.VAL.all_F_Xlr_time:
            mIoU, acc, F_Xlr_all, F_Xlr_score_flat_all = evaluate(segmentation_module, loader_val, cfg, gpu, foveation_module)
        else:
            mIoU, acc, F_Xlr, F_Xlr_score_flat = evaluate(segmentation_module, loader_val, cfg, gpu, foveation_module)
    else:
        mIoU, acc = evaluate(segmentation_module, loader_val, cfg, gpu)


    print('Evaluation Done!')
    if cfg.MODEL.foveation:
        if cfg.VAL.all_F_Xlr_time:
            # print('=============F_Xlr_score_flat_all================\n', F_Xlr_score_flat_all.shape())
            return mIoU, acc, F_Xlr_all, F_Xlr_score_flat_all
        else:
            return mIoU, acc, F_Xlr, F_Xlr_score_flat
    else:
        return mIoU, acc

def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    # crit = nn.NLLLoss(ignore_index=-1)
    # Gleason2019:
    # NOTE: DON'T use ignore_index to omit class 3 which will lead final layer size missmatch, use weight=0 for class 3
    # ignore_label = -1 # because we added 1, so the original gs2 class labelled as 3
    # total_lab_weight [2.0343, 15.8754, inf, 5.2565, 4.0280, 561.1551, 194.2561], inf will be omit by pass 0 instead of inf in the Tensor
    # TODO: weight now calculated based on 67 STAPLE fused gt subset, full
    # class_weights = torch.cuda.FloatTensor([2.0343, 15.8754, 0, 5.2565, 4.0280, 561.1551, 194.2561])
    # omit background and set upper cap as 10
    # class_weights = torch.cuda.FloatTensor([1, 1, 0, 1, 1, 0, 0])
    # crit = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_label)
    if 'CITYSCAPES' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=19)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=19)
    elif 'Digest' in cfg.DATASET.root_dataset:
        if cfg.TRAIN.loss_fun == 'NLLLoss':
            crit = nn.NLLLoss(ignore_index=-2)
        else:
            crit = nn.CrossEntropyLoss(ignore_index=-2)
    elif cfg.TRAIN.loss_fun == 'FocalLoss' and 'DeepGlob' in cfg.DATASET.root_dataset:
        crit = FocalLoss(gamma=6, ignore_label=cfg.DATASET.ignore_index)
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

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        cfg)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/foveation-cityscape-hrnetv2.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
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

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu)
