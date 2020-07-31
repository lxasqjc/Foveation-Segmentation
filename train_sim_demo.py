# General libs
import os
import torch
import torch.nn as nn
# Our libs
from config import cfg
from dataset import TrainDataset, b_imresize
from models import ModelBuilder, SegmentationModule, FovSegmentationModule
from train_fove import checkpoint_last, train, create_optimizers
from eval import eval_during_train


def main(cfg, gpus):
    if 'CITYSCAPE' in cfg.DATASET.list_train:
        crit = nn.NLLLoss(ignore_index=19)
    else:
        crit = nn.NLLLoss(ignore_index=-2)
    # Segmentation Network Builders
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
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit, cfg)
    segmentation_module.cuda()
    nets = (net_encoder, net_decoder, crit)
    # Foveation Network Builders
    if cfg.MODEL.foveation:
        net_foveater = ModelBuilder.build_foveater(
            in_channel=cfg.MODEL.in_dim,
            out_channel=len(cfg.MODEL.patch_bank),
            len_gpus=len(gpus),
            weights=cfg.MODEL.weights_foveater,
            cfg=cfg)
        foveation_module = FovSegmentationModule(net_foveater, cfg, len_gpus=len(gpus))
        foveation_module.cuda()
        nets = (net_encoder, net_decoder, crit, net_foveater)
    # Set up optimizers
    optimizers = create_optimizers(nets, cfg)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # customerized pre-batched dataset
        pin_memory=True)

    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
    # create loader iterator
    iterator_train = iter(loader_train)
    # Main loop
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        if cfg.MODEL.foveation:
            train(segmentation_module, iterator_train,
            optimizers, epoch+1, cfg, history=None, foveation_module=foveation_module)
        else:
            train(segmentation_module, iterator_train,
            optimizers, epoch+1, cfg)
        # save checkpoint
        checkpoint_last(nets, cfg, epoch+1)
        # eval during train
        if cfg.MODEL.foveation:
            val_iou, val_acc, F_Xlr, F_Xlr_score_flat = eval_during_train(cfg)
        else:
            val_iou, val_acc = eval_during_train(cfg)
    print('Training Done!')



cfg.merge_from_file("config/foveation-cityscape-sim-demo.yaml")
if not os.path.isdir(cfg.DIR):
    os.makedirs(cfg.DIR)
gpus = [0] # simple demo only support 1 gpu
main(cfg, gpus)
