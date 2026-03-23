#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""
import os
import torch.utils.data
from torchvision import models
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

# 当前文件所在目录: ciagan-master/source
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录: ciagan-master
_ROOT_DIR = os.path.dirname(_CUR_DIR)
# arcface 目录: ciagan-master/pretrained/arcface-iresnet50
_ARCFACE_DIR = os.path.join(_ROOT_DIR, 'pretrained', 'arcface-iresnet50')

if _ARCFACE_DIR not in sys.path:
    sys.path.insert(0, _ARCFACE_DIR)

from nets.mobilefacenet import MobileFaceNet


# class GANLoss(nn.Module):
#     """
#     兼容老调用方式：
#     - 判别器：
#         loss_D = gan(pred_real, True) + gan(pred_fake, False)   # 对 hinge 生效
#     - 生成器：
#         loss_G_gan = gan.g_loss(pred_fake)                      # hinge 专用入口
#     其它模式(lsgan/vanilla/wgangp)保持原行为。
#     """
#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         super().__init__()
#         self.gan_mode = gan_mode
#         self.register_buffer('real_label', torch.tensor(target_real_label, dtype=torch.float))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label, dtype=torch.float))
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode in ['hinge', 'wgangp']:
#             self.loss = None
#         else:
#             raise NotImplementedError(f'gan mode {gan_mode} not implemented')
#
#     def _collect_logits(self, pred):
#         # 统一成 [logit1, logit2, ...]；兼容 Tensor / [Tensor] / [feat..., logit]
#         if isinstance(pred, (list, tuple)):
#             outs = []
#             for p in pred:
#                 outs.append(p[-1] if isinstance(p, (list, tuple)) else p)
#             return outs
#         return [pred]
#
#     def __call__(self, prediction, target_is_real: bool):
#         logits = self._collect_logits(prediction)
#
#         if self.gan_mode == 'lsgan':
#             loss = 0.0
#             for pred in logits:
#                 tgt = (self.real_label if target_is_real else self.fake_label).expand_as(pred)
#                 loss += self.loss(pred.float(), tgt)
#             return loss / len(logits)
#
#         if self.gan_mode == 'vanilla':
#             loss = 0.0
#             for pred in logits:
#                 tgt = (self.real_label if target_is_real else self.fake_label).expand_as(pred)
#                 loss += self.loss(pred.float(), tgt)
#             return loss / len(logits)
#
#         if self.gan_mode == 'wgangp':
#             loss = 0.0
#             for pred in logits:
#                 loss += (-pred.mean() if target_is_real else pred.mean())
#             return loss / len(logits)
#
#         if self.gan_mode == 'hinge':
#             # 判别器用：True-> ReLU(1 - D(real))；False-> ReLU(1 + D(fake))
#             loss = 0.0
#             for pred in logits:
#                 if target_is_real:
#                     loss += F.relu(1.0 - pred).mean()
#                 else:
#                     loss += F.relu(1.0 + pred).mean()
#             return loss / len(logits)
#
#     # 生成器专用入口（hinge）：-E[D(fake)]
#     def g_loss(self, prediction):
#         assert self.gan_mode == 'hinge', "g_loss() is only for 'hinge'"
#         logits = self._collect_logits(prediction)
#         loss = 0.0
#         for pred in logits:
#             loss += (-pred.mean())
#         return loss / len(logits)


# class GANLoss(nn.Module):
#     """Define different GAN objectives.
#     The GANLoss class abstracts away the need to create the target label tensor
#     that has the same size as the input.
#     """
#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         """ Initialize the GANLoss class.
#         Parameters:
#             gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgan.
#             target_real_label (bool) - - label for a real image
#             target_fake_label (bool) - - label of a fake image
#         Note: Do not use sigmoid as the last layer of Discriminator.
#         LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
#         """
#         super(GANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         self.gan_mode = gan_mode
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode in ['wgangp']:
#             self.loss = None
#         else:
#             raise NotImplementedError('gan mode %s not implemented' % gan_mode)
#
#     def get_target_tensor(self, prediction, target_is_real):
#         """Create label tensors with the same size as the input.
#         Parameters:
#             prediction (tensor) - - tpyically the prediction from a discriminator
#             target_is_real (bool) - - if the ground truth label is for real images or fake images
#         Returns:
#             A label tensor filled with ground truth label, and with the size of the input
#         """
#
#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(prediction)
#
#     def __call__(self, prediction, target_is_real):
#         """Calculate loss given Discriminator's output and grount truth labels.
#         Parameters:
#             prediction (tensor) - - tpyically the prediction output from a discriminator
#             target_is_real (bool) - - if the ground truth label is for real images or fake images
#         Returns:
#             the calculated loss.
#         """
#         if self.gan_mode in ['lsgan', 'vanilla']:
#             if isinstance(prediction, list):
#                 loss = 0
#                 for pred in prediction:
#                     target_tensor = self.get_target_tensor(pred, target_is_real)
#                     loss += self.loss(pred, target_tensor)
#                 loss = loss/len(prediction)
#             else:
#                 target_tensor = self.get_target_tensor(prediction, target_is_real)
#                 loss = self.loss(prediction, target_tensor)
#         elif self.gan_mode == 'wgangp':
#             if target_is_real:
#                 loss = -prediction.mean()
#             else:
#                 loss = prediction.mean()
#         return loss
class GANLoss(nn.Module):
    """GAN 损失，兼容判别器多尺度/多层（嵌套 list/tuple）输出。"""
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode not in ['lsgan', 'vanilla', 'wgangp']:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    # ---- 递归提取各尺度的 logits（若某尺度是 [feat..., logits]，取最后一个）----
    def _extract_logits_list(self, pred):
        out = []
        if torch.is_tensor(pred):
            out.append(pred)
        elif isinstance(pred, (list, tuple)):
            for p in pred:
                if isinstance(p, (list, tuple)):
                    # 典型结构：[feat1, feat2, logits] -> 取最后一个继续递归（以防还有嵌套）
                    out.extend(self._extract_logits_list(p[-1]))
                else:
                    out.extend(self._extract_logits_list(p))
        else:
            raise TypeError(f"Unsupported prediction type: {type(pred)}")
        return out

    def get_target_tensor(self, prediction, target_is_real):
        """prediction 必须是 Tensor（已由 _extract_logits_list 处理）。"""
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    # ---- 自己实现 "mean" 形式的 MSE loss ----
    def _mse_mean(self, input, target):
        # (input - target)^2 的 batch 平均
        return (input - target).pow(2).mean()

    # ---- 自己实现 "mean" 形式的 BCE-with-logits loss ----
    def _bce_logits_mean(self, logits, target):
        # target 是 0/1 的 float
        # loss = -[ y * log σ(x) + (1-y) * log (1-σ(x)) ]
        #       = - y * logσ(x) - (1-y) * logσ(-x)
        pos = - target * F.logsigmoid(logits)
        neg = - (1.0 - target) * F.logsigmoid(-logits)
        return (pos + neg).mean()

    def forward(self, prediction, target_is_real):
        # 统一把判别器输出整理成 [Tensor, Tensor, ...]
        logits_list = self._extract_logits_list(prediction)
        assert len(logits_list) > 0, "No logits extracted from discriminator output."

        if self.gan_mode in ['lsgan', 'vanilla']:
            losses = []
            for logits in logits_list:
                target_tensor = self.get_target_tensor(logits, target_is_real)

                if self.gan_mode == 'lsgan':
                    loss = self._mse_mean(logits, target_tensor)
                else:  # 'vanilla'
                    loss = self._bce_logits_mean(logits, target_tensor)

                losses.append(loss)

            return sum(losses) / len(losses)

        elif self.gan_mode == 'wgangp':
            # WGAN-GP: D(real)-> -E[logits], D(fake)-> +E[logits]
            means = [t.mean() for t in logits_list]
            m = sum(means) / len(means)
            return -m if target_is_real else m
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
            *real_data.shape)
        alpha = alpha.to(device)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
    return gradient_penalty


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class VGGPerceptualLoss(nn.Module):
    """
    支持：
    - 同时使用 source / target 两个参考
    - 通过 pp 控制高层语义更偏向 source 还是 target
    - 仍然只在 mask 区域内计算
    """
    def __init__(self, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        self.device = device

        # 载入 VGG19 预训练模型
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # relu_8 -> relu2_2, relu_16 -> relu3_3, relu_25 -> relu4_3
        self.selected_layers = {
            '8':  'relu2_2',   # 浅层: 纹理、边缘
            '16': 'relu3_3',   # 中层: 形状、局部结构
            '25': 'relu4_3'    # 高层: 语义/身份相关
        }
        self.layer_weights = {
            'relu2_2': 0.5,
            'relu3_3': 1.0,
            'relu4_3': 0.25
        }

        # 哪些层只对齐 source，哪些层参与 source/target 混合
        self.low_mid_layers = {'relu2_2', 'relu3_3'}  # 只贴 source
        self.high_layers    = {'relu4_3'}             # 用 pp 在 src/tgt 之间插值

        self.criterion = nn.L1Loss()
        self.vgg = vgg

    def extract_features(self, x: torch.Tensor):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features

    def forward(self,
                x_fake: torch.Tensor,
                x_src:  torch.Tensor,
                x_tgt:  torch.Tensor = None,
                mask:   torch.Tensor = None,
                pp:     float        = 0.8):
        """
        x_fake: [B,3,H,W]  生成图
        x_src : [B,3,H,W]  原图 (source)
        x_tgt : [B,3,H,W]  目标图 (target)，可为 None
        mask  : [B,1,H,W]  1=脸区域 / 需要约束的区域
        pp    : [0,1]      隐私控制系数: 越大越偏向 target（高层）
        """
        # 安全防御：没有 target 就退化为只对齐 source
        use_tgt = (x_tgt is not None) and (pp > 1e-6)

        feat_fake = self.extract_features(x_fake)
        feat_src  = self.extract_features(x_src)
        feat_tgt  = self.extract_features(x_tgt) if use_tgt else None

        loss = 0.0
        for key, w in self.layer_weights.items():
            f_fake = feat_fake[key]
            f_src  = feat_src[key]

            if use_tgt and key in self.high_layers:
                # 高层：在 source / target 之间插值
                f_tgt = feat_tgt[key]
                target_feat = (1.0 - pp) * f_src + pp * f_tgt
            else:
                # 低 / 中层：完全对齐原图（保证纹理、清晰度）
                target_feat = f_src

            if mask is not None:
                m = mask
                # 允许 mask 是 [B,3,H,W]，压成 1 通道
                if m.size(1) != 1:
                    m = m.mean(dim=1, keepdim=True)  # 或用 sum 也行

                # 插值到特征图大小
                m = F.interpolate(m, size=f_fake.shape[2:],
                                  mode='bilinear', align_corners=False)

                # 扩展到 C 通道
                m = m.repeat(1, f_fake.size(1), 1, 1)

                loss_layer = self.criterion(f_fake * m, target_feat * m)
            else:
                loss_layer = self.criterion(f_fake, target_feat)

            loss += w * loss_layer

            loss += w * loss_layer

        return loss
# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, device='cuda'):
#         super(VGGPerceptualLoss, self).__init__()
#         self.device = device
#
#         # 载入 VGG19 预训练模型
#         vgg = models.vgg19(pretrained=True).features.to(device).eval()
#         for p in vgg.parameters():
#             p.requires_grad = False
#
#         # relu_16 对应 relu3_3, relu_25 对应 relu4_3
#         self.selected_layers = {
#             '8': 'relu2_2',  # 浅层: 纹理、边缘
#             '16': 'relu3_3',  # 中层: 形状、局部结构
#             '25': 'relu4_3'  # 高层: 语义一致性
#         }
#         self.layer_weights = {
#             'relu2_2': 0.5,
#             'relu3_3': 1,
#             'relu4_3': 0.25
#         }
#
#         self.criterion = nn.L1Loss()
#
#         self.vgg = vgg
#
#     def extract_features(self, x):
#         features = {}
#         for name, layer in self.vgg._modules.items():
#             x = layer(x)
#             if name in self.selected_layers:
#                 features[self.selected_layers[name]] = x
#         return features
#
#     def forward(self, x_fake, x_real, mask):
#         """
#         x_fake, x_real: [B,3,H,W]
#         mask: [B,1,H,W], 单通道，值为 0/1 或 0~1
#         """
#         # 提取特征
#         feat_fake = self.extract_features(x_fake)
#         feat_real = self.extract_features(x_real)
#
#         loss = 0.0
#         for key, w in self.layer_weights.items():
#             f_fake = feat_fake[key]
#             f_real = feat_real[key]
#             # 插值 mask 到特征层的大小
#             m = F.interpolate(mask, size=f_fake.shape[2:], mode='bilinear', align_corners=False)
#             # 扩展 mask 使其通道数与特征图相同（重复mask的单通道）
#             m = m.sum(dim=1, keepdim=True)
#             m = m.repeat(1, f_fake.size(1), 1, 1)  # 扩展 mask 的通道数
#             # 计算加权 L1 损失，仅在 mask 区域内
#             loss += w * self.criterion(f_fake * m, f_real * m)
#
#         return loss

class ArcFaceBackbone(nn.Module):
    """
    只负责输出 [B, 512] 的身份 embedding，参数冻住，但对输入保留梯度。
    这里假设你已经有 iresnet50 + arcface 的实现，对应的 state_dict
    和这份模型结构是一致的。
    """
    def __init__(self, backbone, device='cuda'):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False  # 冻住权重，只对输入求梯度

    def forward(self, x):
        # x: [B, 3, H, W]，假设在 [0, 1] 范围
        # 1) resize 到 ArcFace 训练用的 112x112
        x = F.interpolate(x, size=(112, 112),
                          mode='bilinear', align_corners=False)

        # 2) 按 arcface-pytorch 的习惯做归一化: (img - 127.5) / 128.0
        # 如果你的训练图像是 [-1,1]，这里要改成先反归一化。
        x = x * 255.0
        x = (x - 127.5) / 128.0

        # 3) backbone forward，返回 [B, 512]，一般已经 L2-normalized，
        # 如果没有，可以在外面再做一次 F.normalize
        feat = self.backbone(x)
        feat = F.normalize(feat, dim=1)
        return feat


# util_loss.py 里面
def build_arcface_backbone(weight_path, device='cuda'):
    """
    使用你当前工程里的 MobileFaceNet + ArcFace 预训练权重
    来构建一个身份特征提取器。
    """
    # 这里用的是 pretrained/arcface-iresnet50/nets/mobilefacenet.py 里的类/函数
    # from nets.mobilefacenet import MobileFaceNet  # 名字以文件里为准

    # 1) 实例化结构 —— 打开 mobilefacenet.py 看一下，如果是 class MobileFaceNet(nn.Module): 就这么写
    backbone = MobileFaceNet(embedding_size=128)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # === 2) 加载权重 ===
    state = torch.load(weight_path, map_location=device)

    # 兼容 {"state_dict": ...} 这种格式
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    cleaned_state = {}
    for k, v in state.items():
        name = k

        # 先去掉最外层的 'arcface.' 前缀
        if name.startswith('arcface.'):
            name = name[len('arcface.'):]  # 去掉 'arcface.'

        # 再去掉可能存在的 'module.' 前缀（多卡训练）
        if name.startswith('module.'):
            name = name[len('module.'):]

        cleaned_state[name] = v

    return ArcFaceBackbone(backbone, device=device)



class IDLoss(nn.Module):
    """
    用 ArcFace 特征做身份损失：
      - 拉近 fake 与 target 身份
      - 推远 fake 与 source 身份（可选）
    先不强行加入 pp，有了基础版本再往里加 pp 插值会比较容易。
    """
    def __init__(self, id_backbone, lambda_pull=1.0, lambda_push=0.0, margin=0.3):
        super().__init__()
        self.id_backbone = id_backbone
        self.lambda_pull = lambda_pull   # fake -> target
        self.lambda_push = lambda_push   # fake 离 source 远一些
        self.margin = margin

    def forward(self, img_src, img_tgt, img_fake):
        """
        img_src: [B,3,H,W]  原图（source identity）
        img_tgt: [B,3,H,W]  目标身份图
        img_fake:[B,3,H,W]  生成图（output_gen）
        """
        # 提前处理：只需要 src/tgt 的特征值不反向传播
        with torch.no_grad():
            f_src = self.id_backbone(img_src)   # [B,512]
            f_tgt = self.id_backbone(img_tgt)   # [B,512]

        # fake 这条需要梯度，不能加 no_grad
        f_fake = self.id_backbone(img_fake)     # [B,512]

        # cos 相似度越大越像
        cos_fake_tgt = F.cosine_similarity(f_fake, f_tgt, dim=1)
        cos_fake_src = F.cosine_similarity(f_fake, f_src, dim=1)

        # 让 fake 尽量接近 target：1 - cos(fake, tgt)
        loss_pull = (1.0 - cos_fake_tgt).mean()

        # 让 fake 和 source 保持一定角度：max(0, cos(fake, src) - margin)
        # margin 越小要求越严格
        if self.lambda_push > 0:
            loss_push = F.relu(cos_fake_src - self.margin).mean()
        else:
            loss_push = torch.zeros_like(loss_pull)

        loss = self.lambda_pull * loss_pull + self.lambda_push * loss_push
        return loss, {
            "loss_pull": loss_pull.detach(),
            "loss_push": loss_push.detach(),
            "cos_fake_tgt": cos_fake_tgt.mean().detach(),
            "cos_fake_src": cos_fake_src.mean().detach(),
        }

class ColorConsistencyLoss(nn.Module):
        """
        只在低频上约束颜色/光照的一致性：
        - 先做较大核的平均池化，相当于模糊掉高频细节；
        - 再在 mask 区域做 L1。
        """

        def __init__(self, kernel_size=21):
            super().__init__()
            # kernel_size 可以 15/21 之类，越大越偏向“整体颜色”
            self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

        def forward(self, x_fake, x_real, mask):
            """
            x_fake, x_real: [B, 3, H, W]
            mask:          [B, 1, H, W]  (人脸区域 = 1)
            """
            # 1) 低频图（模糊）
            lf_fake = self.pool(x_fake)
            lf_real = self.pool(x_real)

            # 2) mask broadcast 到 3 通道
            if mask.size(1) == 1:
                m = mask.repeat(1, 3, 1, 1)
            else:
                m = mask

            # 3) 只在 mask 内比较低频颜色
            diff = (lf_fake - lf_real).abs() * m
            denom = m.sum() * x_fake.shape[1] + 1e-8
            loss = diff.sum() / denom
            return loss