from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.list_test = ""
_C.DATASET.class_mapping = 0
_C.DATASET.ignore_index = -2
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = "Flip"
_C.DATASET.multi_scale_aug = False
_C.DATASET.adjust_crop_range = False
_C.DATASET.mirror_padding = False

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048

# foveation implementation
# Note currently only support training set with identical size/shape
_C.MODEL.foveation = False
_C.MODEL.hard_fov = False
_C.MODEL.hard_grad = 'st_inv' # default straight through + weighed inversely proportionally to the probability of sampling
# ***central crop pred of hard selected patch with identical area of P1
_C.MODEL.hard_fov_pred = False
_C.MODEL.categorical = False # stochastic gradient estimators via torch categorical
_C.MODEL.inv_categorical = False
_C.MODEL.gumbel_softmax = False
_C.MODEL.gumbel_softmax_st = False
_C.MODEL.gumbel_tau = 1.0
_C.MODEL.gumbel_tau_anneal = False
# int select single patch size rather than concatenated tnsor of all patch sizes
_C.MODEL.hard_select = False
_C.MODEL.fov_padding = True
_C.MODEL.deep_fov = ''
_C.MODEL.fov_normalise = ''
_C.MODEL.fov_activation = 'relu'
# Force Gradient from Foveation output to zero at each step
_C.MODEL.Zero_Step_Grad = False
# input image channel
_C.MODEL.in_dim = 3
# downsample rate for Xlr
_C.MODEL.fov_map_scale = 100
# multiscale patch size bank
_C.MODEL.patch_bank = [1100, 2200, 4400]
_C.MODEL.patch_ap = 1
# setting for one-hot fixed patch size baselines
_C.MODEL.one_hot_patch = []
_C.MODEL.weights_foveater = ""
_C.MODEL.weights_fov_res = ""
_C.MODEL.pre_cropped = False
_C.MODEL.cropped_lists = []

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.auto_batch = 'auto10'
_C.TRAIN.gpu_threshold = 0.65e6
_C.TRAIN.batch_size_per_gpu = 2
# number of iterations per batch
_C.TRAIN.fov_location_step = 1
# default fov_location_step = num of pixels in Xlr
_C.TRAIN.auto_fov_location_step = False
_C.TRAIN.sync_location = 'mean_mbs'
_C.TRAIN.mini_batch_size = 1
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
_C.TRAIN.loss_fun = ""
_C.TRAIN.optim = "SGD"
_C.TRAIN.fov_scale_pow = 1 # scale distribution of lr_scale/wd_scale
# ***use fov average patch size to scale learning rate (ini imp at 4th Mar 2020)
_C.TRAIN.fov_scale_lr = ''
# ***use fov average patch size to scale weight decay (ini imp at 4th Mar 2020)
_C.TRAIN.fov_scale_weight_decay = ''
_C.TRAIN.fov_scale_seg_only = False
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
_C.TRAIN.lr_foveater = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.weight_decay_fov = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304
# whether save checkpoint of last epoch
_C.TRAIN.save_checkpoint = True
# number of epochs to perform eval_during_train
_C.TRAIN.eval_per_epoch = 1
# number of epochs to save checkpoints
_C.TRAIN.checkpoint_per_epoch = 2000

# entropy regularisation
_C.TRAIN.entropy_regularisation = False
_C.TRAIN.entropy_regularisation_weight = 1.0

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"
# visualize hard max fov maps
_C.VAL.hard_max_fov = False
_C.VAL.max_score = False
_C.VAL.central_crop = False
# save F_Xlr_time for all val data
_C.VAL.all_F_Xlr_time = False
_C.VAL.rename_eval_folder = ""
_C.VAL.multipro = False # currently not supported
_C.VAL.dice = False
_C.VAL.hd95 = False
_C.VAL.F_Xlr_only = False
_C.VAL.F_Xlr_acc_map_only = False
_C.VAL.foveated_expection = True
# option to output score_maps of fixed patch baselines for later ensemble, currently used
_C.VAL.ensemble = False
# option to ensemble fixed patch baselines, NEED TEST
_C.VAL.approx_pred_Fxlr_by_ensemble = False
# downsample F_Xlr for efficient inference
_C.VAL.F_Xlr_low_scale = 0
_C.VAL.expand_prediection_rate = 1 # =2 for HRnet in cityscapes
_C.VAL.expand_prediection_rate_patch = 1.0 # =2 for HRnet in cityscapes

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"
