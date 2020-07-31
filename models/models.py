import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import torchvision
from . import resnet, resnext, mobilenet, hrnet, u_net, attention_u_net, attention_u_net_deep, attention_u_net_deep_ds4x, hrnetv2_nonsyn
from lib.nn import SynchronizedBatchNorm2d
from dataset import imresize, b_imresize, patch_loader

BatchNorm2d = SynchronizedBatchNorm2d
BN_MOMENTUM = 0.1

class FovResModule(nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        # in_channels: num of channels corresponds to input image channels, e.g. 3
        # out_channels: num of channels corresponds to num of sclaes tested
        super(FovResModule, self).__init__()
        self.compress = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(cfg.MODEL.patch_bank[0] // cfg.DATASET.segm_downsampling_rate)

    def forward(self, x):
        return self.pool(self.compress(x))

class FoveationModule(nn.Module):
    def __init__(self, in_channels, out_channels, len_gpus, cfg):
        # in_channels: num of channels corresponds to input image channels, e.g. 3
        # out_channels: num of channels corresponds to num of sclaes tested
        super(FoveationModule, self).__init__()
        self.cfg = cfg
        self.fov_expand_1 = nn.Conv2d(in_channels=in_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_expand_2 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_squeeze_1 = nn.Conv2d(in_channels=8*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

        if cfg.MODEL.fov_normalise == 'instance':
            self.norm1 = nn.InstanceNorm2d(8*out_channels, momentum=BN_MOMENTUM, affine=True)
            self.norm2 = nn.InstanceNorm2d(8*out_channels, momentum=BN_MOMENTUM, affine=True)
            self.norm3 = nn.InstanceNorm2d(out_channels, momentum=BN_MOMENTUM, affine=True)
        else:
            # bn
            self.norm1 = BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
            self.norm2 = BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
            self.norm3 = BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        if cfg.MODEL.fov_activation == 'relu':
            self.act = nn.ReLU6(inplace=True)
        elif cfg.MODEL.fov_activation == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)

        self.acc_grad = [[] for _ in range(len_gpus)]
        # for tensorboard gradient visualization
        self.save_print_grad = [{'layer1_grad': None, 'layer2_grad': None, 'layer3_grad': None} for _ in range(len_gpus)]

        if self.cfg.MODEL.deep_fov == 'deep_fov1':
            self.fov_expand_3 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
            self.fov_expand_4 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
        # self.requires_grad = True
        if self.cfg.MODEL.deep_fov == 'deep_fov2':
            self.fov_expand_2 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=5, padding=2, bias=False)
            self.fov_squeeze_1 = nn.Conv2d(in_channels=8*out_channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False)

    def forward(self, x, reset_grad=True, train_mode=True):
        # TODO: F.softplus(Ci) / Sum(C[:])
        if self.cfg.MODEL.deep_fov == 'deep_fov1':
            output = F.softmax(self.norm3(self.fov_squeeze_1(self.act(self.fov_expand_4(self.act(self.fov_expand_3(self.act(self.norm2(self.fov_expand_2(self.act(self.norm1(self.fov_expand_1(x)))))))))))), dim=1)
        else:
            if self.cfg.MODEL.fov_normalise == 'no_normalise':
                layer3 = self.fov_squeeze_1(self.act(self.fov_expand_2(self.act(self.fov_expand_1(x)))))
            elif self.cfg.MODEL.fov_normalise == 'bn1':
                layer3 = self.fov_squeeze_1(self.act(self.fov_expand_2(self.act(self.norm1(self.fov_expand_1(x))))))
            else:
                if train_mode:
                    layer1 = self.act(self.norm1(self.fov_expand_1(x)))
                    layer1.register_hook(self.print_grad_layer1)
                    layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
                    layer2.register_hook(self.print_grad_layer2)
                    layer3 = self.norm3(self.fov_squeeze_1(layer2))
                    layer3.register_hook(self.print_grad_layer3)
                else:
                    layer1 = self.act(self.norm1(self.fov_expand_1(x)))
                    layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
                    layer3 = self.norm3(self.fov_squeeze_1(layer2))
            if self.cfg.MODEL.gumbel_softmax:
                output = F.log_softmax(layer3, dim=1)
            else:
                output = F.softmax(layer3, dim=1)
        if train_mode:
            if reset_grad:
                output.register_hook(self.save_grad)
            else:
                output.register_hook(self.manipulate_grad)
        if train_mode and not reset_grad:
            return output, self.save_print_grad[0]
        else:
            return output

    def print_grad_layer1(self, grad):
        self.save_print_grad[grad.device.index]['layer1_grad'] = grad.clone()
    def print_grad_layer2(self, grad):
        self.save_print_grad[grad.device.index]['layer2_grad'] = grad.clone()
    def print_grad_layer3(self, grad):
        self.save_print_grad[grad.device.index]['layer3_grad'] = grad.clone()

    def save_grad(self, grad):
        self.acc_grad[grad.device.index].append(grad.clone())
        if self.cfg.MODEL.Zero_Step_Grad:
            grad *= 0

    def manipulate_grad(self, grad):
        self.acc_grad[grad.device.index].append(grad.clone())
        total_grad = torch.cat(self.acc_grad[grad.device.index])
        total_grad = torch.sum(total_grad, dim=0)
        grad.data += total_grad.data
        self.acc_grad[grad.device.index] = []


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class FovSegmentationModule(SegmentationModuleBase):
    def __init__(self, net_foveater, cfg, len_gpus=2):
        super(FovSegmentationModule, self).__init__()
        self.foveater = net_foveater
        self.cfg = cfg
        self.F_Xlr_i_soft = [None for _ in range(len_gpus)]
        self.F_Xlr_i_grad = [None for _ in range(len_gpus)]

    def forward(self, batch_data, *, train_mode=True):
        # print('in device', batch_data['img_data'].device)
        if train_mode:
            xi, yi, rand_location, fov_location_batch_step = batch_data['cor_info']
        else:
            xi, yi = batch_data['cor_info']
        X, Y = batch_data['img_data'], batch_data['seg_label']
        if not train_mode:
            X_unnorm = batch_data['img_data_unnorm']
        fov_map_scale = self.cfg.MODEL.fov_map_scale
        X_lr = b_imresize(X, (round(X.shape[2]/fov_map_scale), round(X.shape[3]/(fov_map_scale*self.cfg.MODEL.patch_ap))), interp='bilinear')
        if self.cfg.MODEL.one_hot_patch != []:
            if max(self.cfg.MODEL.one_hot_patch) == 0:
                one_hot_patch_temp = self.cfg.MODEL.one_hot_patch
                one_hot_patch_temp[random.randint(0,len(one_hot_patch_temp)-1)] = 1
                one_hot_tensor = torch.FloatTensor(one_hot_patch_temp)
            else:
                one_hot_tensor = torch.FloatTensor(self.cfg.MODEL.one_hot_patch)
            one_hot_tensor = one_hot_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            F_Xlr = one_hot_tensor.expand((X_lr.shape[0],len(self.cfg.MODEL.patch_bank),X_lr.shape[2],X_lr.shape[3])).to(X_lr.device)
            print_grad = torch.zeros(F_Xlr.shape).to(X_lr.device)

        else:
            if train_mode:
                if fov_location_batch_step == rand_location:
                    F_Xlr, print_grad = self.foveater(X_lr, reset_grad=False, train_mode=True) # b,d,w,h
                else:
                    F_Xlr = self.foveater(X_lr, reset_grad=True, train_mode=True) # b,d,w,h
            else:
                F_Xlr = self.foveater(X_lr, reset_grad=False, train_mode=False)
        if self.cfg.VAL.F_Xlr_low_scale != 0 and 'F_Xlr_low_res' in batch_data:
            F_Xlr = batch_data['F_Xlr_low_res']
        F_Xlr_i = F_Xlr[:,:,xi,yi] # b,d,m    m-> mini_batch size len(xi)
        hard_selected_scale = None
        if self.cfg.MODEL.hard_fov:
            if train_mode:
                self.F_Xlr_i_soft[F_Xlr_i.device.index] = F_Xlr_i
            if train_mode and self.cfg.MODEL.one_hot_patch == []:
                F_Xlr_i.register_hook(self.modify_argmax_grad)
            _, max_idx = torch.max(F_Xlr_i, dim=1)

            if self.cfg.MODEL.hard_fov_pred:
                patch_data['hard_max_idx'] = max_idx

            mask = (torch.ones(F_Xlr_i.shape).long()*torch.arange(F_Xlr_i.size(1)).reshape(F_Xlr_i.size(1),-1)).to(F_Xlr_i.device) == max_idx.unsqueeze(1).to(F_Xlr_i.device)
            # mask = torch.arange(F_Xlr_i.size(1)).reshape(1,-1).to(F_Xlr_i.device) == max_idx.unsqueeze(1).to(F_Xlr_i.device)
            # important trick: normal inplace operation like a[mask] = 1 will eturns a broken gradient, use tensor.masked_fill_
            # https://github.com/pytorch/pytorch/issues/1011
            F_Xlr_i_hard = F_Xlr_i.clone().masked_fill_(mask.eq(0), 0).masked_fill_(mask.eq(1), 1)

            if train_mode and self.cfg.MODEL.one_hot_patch == []:
                F_Xlr_i_hard.register_hook(self.hook_F_Xlr_i_grad)

            # print(max_idx.size())
            if self.cfg.MODEL.hard_select:
                hard_selected_scale = max_idx
        elif self.cfg.MODEL.categorical:
            # patch_probs = foveation_net(x_lr) # get the probabilities over patches from FoveationModule
            # mbs > 1 not currently supported
            m = Categorical(F_Xlr_i.permute(0,2,1)) # categorical see last dimension as prob
            patch_selected = m.sample() # select a patch by sampling from the distribution, e.g. tensor(3)
            # mask = torch.arange(F_Xlr_i.size(1)).reshape(1,-1).to(F_Xlr_i.device) == patch_selected.unsqueeze(1).to(F_Xlr_i.device)
            # F_Xlr_i_hard = F_Xlr_i.clone().masked_fill_(mask.eq(0), 0).masked_fill_(mask.eq(1), 1)
            hard_selected_scale = patch_selected
        elif self.cfg.MODEL.gumbel_softmax:
            # print('applied gumbel_tau: ', self.cfg.MODEL.gumbel_tau)
            if self.cfg.MODEL.gumbel_softmax_st:
                # s.t.
                F_Xlr_i_hard = F.gumbel_softmax(F_Xlr_i, tau=self.cfg.MODEL.gumbel_tau, hard=True, dim=1)
            else:
                F_Xlr_i_hard = F.gumbel_softmax(F_Xlr_i, tau=self.cfg.MODEL.gumbel_tau, hard=False, dim=1)

        # if self.cfg.VAL.F_Xlr_only:
        #     hard_selected_scale = torch.tensor([[[0]]])
        patch_data = dict()
        # iter over mini_batch
        for i_mb in range(len(xi)):
            xi_i = xi[i_mb]
            yi_i = yi[i_mb]
            if train_mode:
                if self.cfg.MODEL.categorical or (self.cfg.MODEL.hard_fov and self.cfg.MODEL.hard_select):
                    X_patches, seg_label = patch_loader(X, Y, xi_i, yi_i, self.cfg, train_mode=train_mode, select_scale=hard_selected_scale[:,i_mb])
                else:
                    X_patches, seg_label = patch_loader(X, Y, xi_i, yi_i, self.cfg, train_mode=train_mode, select_scale=hard_selected_scale)
            else:
                if self.cfg.MODEL.categorical or (self.cfg.MODEL.hard_fov and self.cfg.MODEL.hard_select):
                    X_patches, Y_patch_cord, X_patches_cords, seg_label = patch_loader(X, Y, xi_i, yi_i, self.cfg, train_mode=False, select_scale=hard_selected_scale[:,i_mb])
                else:
                    X_patches, Y_patch_cord, X_patches_cords, seg_label = patch_loader(X, Y, xi_i, yi_i, self.cfg, train_mode=False, select_scale=hard_selected_scale)
                if self.cfg.VAL.visualize:
                    if self.cfg.MODEL.categorical or (self.cfg.MODEL.hard_fov and self.cfg.MODEL.hard_select):
                        X_patches_unnorm, Y_patch_cord_unnorm, X_patches_cords_unnorm, seg_label_unnorm = patch_loader(X_unnorm, Y, xi_i, yi_i, self.cfg, train_mode=False, select_scale=hard_selected_scale[:,i_mb])
                    else:
                        X_patches_unnorm, Y_patch_cord_unnorm, X_patches_cords_unnorm, seg_label_unnorm = patch_loader(X_unnorm, Y, xi_i, yi_i, self.cfg, train_mode=False, select_scale=hard_selected_scale)
                    X_patches_unnorm =  torch.cat([item.unsqueeze(0) for item in X_patches_unnorm]) # d,b,c,w,h (item
                    X_patches_unnorm = X_patches_unnorm.permute(1,0,2,3,4) # b,d,c,w,h
            # convert list to tensor
            X_patches =  torch.cat([item.unsqueeze(0) for item in X_patches]) # d,b,c,w,h (item
            X_patches = X_patches.permute(1,0,2,3,4) # b,d,c,w,h


            if self.cfg.MODEL.hard_fov or self.cfg.MODEL.gumbel_softmax:
                if self.cfg.MODEL.hard_select:
                    weighted_average = X_patches[:,0,:,:,:]*(patch_selected/patch_selected).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float() # d = 1 as selected single scale
                else:
                    weighted_patches = F_Xlr_i_hard[:,:,i_mb].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*X_patches.size()).to(X_patches.device)*X_patches
                    weighted_average = torch.sum(weighted_patches, dim=1) # b,c,w,h
            elif self.cfg.MODEL.categorical:
                weighted_average = X_patches[:,0,:,:,:] # b,c,w,h
            else:
                weighted_patches = F_Xlr[:,:,xi_i,yi_i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*X_patches.size()).to(X_patches.device)*X_patches
                weighted_average = torch.sum(weighted_patches, dim=1) # b,c,w,h
            if i_mb == 0:
                seg_label_mb = seg_label
                weighted_average_mb = weighted_average
            else:
                if train_mode:
                    seg_label_mb = torch.cat([seg_label_mb, seg_label])
                weighted_average_mb = torch.cat([weighted_average_mb, weighted_average])
        patch_data['seg_label'] = seg_label_mb
        patch_data['img_data'] = weighted_average_mb
        if self.cfg.MODEL.categorical:
            patch_data['log_prob_act'] = m.log_prob(patch_selected)


        # training
        if train_mode:
            if fov_location_batch_step == rand_location:
                return patch_data, F_Xlr, print_grad
            else:
                return patch_data, F_Xlr
        # inference
        elif self.cfg.VAL.visualize:
            return patch_data, F_Xlr, Y_patch_cord, X_patches_cords, X_patches_unnorm
        else:
            return patch_data, F_Xlr, Y_patch_cord

    def hook_F_Xlr_i_grad(self, grad):
        # print('F_Xlr_i_grad', grad)
        self.F_Xlr_i_grad[grad.device.index] = grad.clone()

    def modify_argmax_grad(self, grad):
        # print('argmax_grad', grad)
        if self.cfg.MODEL.hard_grad == "st_inv":
            self.F_Xlr_i_grad[grad.device.index] /= self.F_Xlr_i_soft[grad.device.index]
        # normal straight though
        # print('ori_argmax_grad:', grad.data)
        grad.data = self.F_Xlr_i_grad[grad.device.index].data
        # print('modifyed_argmax_grad', grad)



class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, cfg, deep_sup_scale=None, net_fov_res=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.net_fov_res = net_fov_res

    # @torchsnooper.snoop()
    def forward(self, feed_dict, *, segSize=None, F_Xlr_acc_map=False):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            elif self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            if self.cfg.MODEL.hard_fov_pred:
                hard_pred = []
                patch_bank = self.cfg.MODEL.patch_bank
                segm_downsampling_rate = self.cfg.DATASET.segm_downsampling_rate
                hard_max_idx = feed_dict['hard_max_idx']
                for b in range(feed_dict['seg_label'].shape[0]):
                    hard_max_patch_size = patch_bank[hard_max_idx[b]]
                    # print('hard_max_idx:', hard_max_idx[b])
                    central_patch_size = patch_bank[0]
                    cx = (hard_max_patch_size//2 - central_patch_size//2) // (hard_max_patch_size//central_patch_size)
                    cy = (hard_max_patch_size//2 - central_patch_size//2) // (hard_max_patch_size//central_patch_size)
                    central_patch_size = central_patch_size // (hard_max_patch_size//central_patch_size)
                    if segm_downsampling_rate != 1:
                        central_patch_size = central_patch_size // segm_downsampling_rate
                        cx = cx // segm_downsampling_rate
                        cy = cy // segm_downsampling_rate
                    central_crop = pred[b][:, cx:cx+central_patch_size, cy:cy+central_patch_size].clone()
                    central_crop = central_crop.unsqueeze(0)
                    # print('central_crop_shape:', central_crop.shape)
                    # print('pred:', pred[b])
                    hard_pred.append(F.interpolate(central_crop, (pred[b].shape[1], pred[b].shape[2]), mode='bilinear').clone())
                    # print('hard_pred:', hard_pred[b])
                hard_pred = torch.cat(hard_pred, dim=0)
                loss = self.crit(hard_pred, feed_dict['seg_label'])
            else:
                loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            if self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize, res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            if self.cfg.MODEL.hard_fov_pred:
                patch_bank = list((float(self.cfg.VAL.expand_prediection_rate_patch)*np.array(self.cfg.MODEL.patch_bank)).astype(int))
                hard_max_idx = feed_dict['hard_max_idx']
                for b in range(feed_dict['img_data'].shape[0]):
                    hard_max_patch_size = patch_bank[hard_max_idx[b]]
                    # print('hard_max_idx:', hard_max_idx[b])
                    central_patch_size = patch_bank[0]
                    cx = (hard_max_patch_size//2 - central_patch_size//2) // (hard_max_patch_size//central_patch_size)
                    cy = (hard_max_patch_size//2 - central_patch_size//2) // (hard_max_patch_size//central_patch_size)
                    central_patch_size = central_patch_size // (hard_max_patch_size//central_patch_size)

                    central_crop = pred[b][:, cx:cx+central_patch_size, cy:cy+central_patch_size].clone()
                    central_crop = central_crop.unsqueeze(0)
                    # print('central_crop_shape:', central_crop.shape)
                    # print('pred:', pred[b])
                    pred[b] = F.interpolate(central_crop, (pred[b].shape[1], pred[b].shape[2]), mode='bilinear')[0].clone()

            if F_Xlr_acc_map:
                loss = self.crit(pred, feed_dict['seg_label'])
                return pred, loss
            else:
                return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50', fc_dim=2048, weights='', dilate_rate=4):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=dilate_rate)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=dilate_rate)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        elif arch == 'hrnetv2_nonsyn':
            net_encoder = hrnetv2_nonsyn.__dict__['hrnetv2_nonsyn'](pretrained=False)
        elif arch == 'hrnetv2_nopretrain':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=False)
        elif arch == 'u_net':
            net_encoder = u_net.__dict__['u_net'](pretrained=pretrained)
        elif arch == 'attention_u_net':
            net_encoder = attention_u_net.__dict__['attention_u_net'](pretrained=pretrained, width=fc_dim)
        elif arch == 'attention_u_net_deep':
            net_encoder = attention_u_net_deep.__dict__['attention_u_net_deep'](pretrained=pretrained)
        elif arch == 'attention_u_net_deep_ds4x':
            net_encoder = attention_u_net_deep_ds4x.__dict__['attention_u_net_deep_ds4x'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='upernet',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_u':
            net_decoder = C1_U(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_hrnet_cityscape':
            net_decoder = C1_hrnet_cityscape(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_8090':
            net_decoder = C1_8090(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1_upsample':
            net_decoder = C1_upsample(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def build_foveater(in_channel=3,
                      out_channel=3,
                      len_gpus=2,
                      weights='',
                      cfg=None):
        net_foveater = FoveationModule(in_channel, out_channel, len_gpus, cfg)

        if len(weights) == 0:
            net_foveater.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_foveater')
            net_foveater.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_foveater

    @staticmethod
    def build_fov_res(in_channel=3,
                      out_channel=1,
                      weights='',
                      cfg=None):
        net_fov_res = FovResModule(in_channel, out_channel, cfg)

        if len(weights) == 0:
            net_fov_res.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_fov_res')
            net_fov_res.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_fov_res


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 2:
            orig_resnet.conv1.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer2.apply(
                partial(self._nostride_dilate, dilate=4))
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=8))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=16))
        elif dilate_scale == 4:
            orig_resnet.layer2.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=4))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=8))
        elif dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1_U(nn.Module):
    def __init__(self, num_class=7, fc_dim=32, use_softmax=False):
        super(C1_U, self).__init__()
        self.use_softmax = use_softmax
        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.conv_last(conv5)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None, res=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if res is not None:
            x += res

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C1_hrnet_cityscape(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1_hrnet_cityscape, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None, res=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if res is not None:
            x += res

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.interpolate(
                x, size=(x.shape[-2]*4, x.shape[-1]*4), mode='bilinear', align_corners=False)
            x = nn.functional.log_softmax(x, dim=1)

        return x

class C1_8090(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1_8090, self).__init__()
        self.use_softmax = use_softmax


    def forward(self, x, segSize=None, res=None):

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=(x.shape[-2]*4, x.shape[-1]*4), mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.interpolate(
                x, size=(x.shape[-2]*4, x.shape[-1]*4), mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)

        return x

class C1_upsample(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1_upsample, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None, res=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if res is not None:
            x += res

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.log_softmax(x, dim=1)
        else:
            x = nn.functional.interpolate(
                x, size=(512, 512), mode='bilinear', align_corners=False)
            x = nn.functional.log_softmax(x, dim=1)

        return x

# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
