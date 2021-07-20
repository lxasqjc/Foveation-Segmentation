# General libs
import os
import torch
import torch.nn as nn
import argparse
import random
import pandas as pd
# Our libs
from config import cfg
from dataset import TrainDataset, b_imresize
from models import ModelBuilder, SegmentationModule, FovSegmentationModule
from train_fove import checkpoint_last, train, create_optimizers
from eval import eval_during_train
from utils import AverageMeter, parse_devices, setup_logger

def checkpoint_history(history, cfg, epoch):
  print('Saving history...')
  # save history as csv
  data_frame = pd.DataFrame(
      data={'train_loss': history['epoch']
          , 'train_loss': history['train_loss']
          , 'train_acc': history['train_acc']
          , 'val_miou': history['val_miou']
          , 'val_acc': history['val_acc']
            }
  )
  data_frame.to_csv('{}/history_epoch_last.csv'.format(cfg.DIR),
                    index_label='epoch')

  torch.save(
      history,
      '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))

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

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_miou': [], 'val_acc': []}
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
    # create loader iterator
    iterator_train = iter(loader_train)
    # Main loop
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        if cfg.MODEL.foveation:
            train_acc, train_loss = train(segmentation_module, iterator_train,
            optimizers, epoch+1, cfg, history=None, foveation_module=foveation_module)
        else:
            train_acc, train_loss = train(segmentation_module, iterator_train,
            optimizers, epoch+1, cfg)
        # save checkpoint
        checkpoint_last(nets, cfg, epoch+1)
        # eval during train
        if cfg.MODEL.foveation:
            val_iou, val_acc, F_Xlr, F_Xlr_score_flat = eval_during_train(cfg)
        else:
            val_iou, val_acc = eval_during_train(cfg)

        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_miou'].append(val_iou)
        history['val_acc'].append(val_acc)
        checkpoint_history(history, cfg, epoch)
    print('Training Done!')

if __name__ == '__main__':

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

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    gpus = [0] # simple demo only support 1 gpu
    main(cfg, gpus)
