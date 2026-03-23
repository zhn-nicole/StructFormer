import copy
from typing import Optional, List
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from torch import nn, Tensor
# from .function import normal,normal_style
# from .function import  calc_mean_std
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.utils as nn_utils
from .ViT_helper import DropPath, to_2tuple, trunc_normal_
import os
from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from . import transformer

# main architecture. use concatenation
class PatchEmbed(nn.Module):

    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)# to_2tuple将输入转换为二元元组
        patch_size = to_2tuple(patch_size) #确保img_size是 (height, width) 格式
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        #定义投影层  X-f+2P/S +1
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #定义上采样层 放大两倍  输出128

    def forward(self, x):
        B, C, H, W = x.shape   #输入（B，3，128，128）
        x = self.proj(x)       #输出（B，512，32，32）

        return x

class SelfAttention2d(nn.Module):
    """
    SAGAN 风格的 2D 自注意力模块：
    输入: [B, C, H, W]
    输出: [B, C, H, W]  (带残差)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels,      kernel_size=1)

        # 可学习的缩放系数，刚开始=0，不会破坏原来结果
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.size()

        # [B, C', H, W] -> [B, C', HW] -> [B, HW, C']
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # Q
        # [B, C', H, W] -> [B, C', HW]
        proj_key   = self.key_conv(x).view(B, -1, H * W)                      # K
        # [B, HW, HW]
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # [B, C, H, W] -> [B, C, HW]
        proj_value = self.value_conv(x).view(B, C, H * W)                     # V
        # [B, C, HW] x [B, HW, HW]^T -> [B, C, HW]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # 残差 + 缩放
        out = self.gamma * out + x
        return out

# class decoder(nn.Module):
#     def __init__(self, img_size=128):
#         super().__init__()
#         self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 32->64
#         self.in4_d = nn.InstanceNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 64->128
#         self.in3_d = nn.InstanceNorm2d(128)
#         self.deconv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.in2_d = nn.InstanceNorm2d(64)
#         self.deconv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.in1_d = nn.InstanceNorm2d(32)
#
#         self.conv_end = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
#         self.img_size = img_size
#
#     def forward(self, x):
#         # x: (B, 512, 32, 32)
#         x = self.in4_d(self.deconv4(x))  # -> (B, 256, 64, 64)
#         x = self.in3_d(self.deconv3(x))  # -> (B, 128, 128, 128)
#         x = self.in2_d(self.deconv2(x))  # -> (B, 64, 128, 128)
#         x = self.in1_d(self.deconv1(x))  # -> (B, 32, 128, 128)
#         x = self.conv_end(x)             # -> (B, 3, 128, 128)
#         return x

vgg = nn.Sequential(
    #假设输入（B, 3, 256, 256)
    nn.Conv2d(3, 3, (1, 1)),#(B, 3, 256, 256)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),#(B, 64, 258, 258)
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),#(B, 64, 260, 260)
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),#(B, 64, 130, 130)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),#(B, 128, 132, 132)
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),#(B, 128, 134, 134)
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),#(B, 128, 67, 67)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),#(B, 256, 75, 75)
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),#(B, 256, 38, 38)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),#(B, 512, 46, 46)
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),#(B, 512, 23, 23)
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
    #输出 (B, 512, 31, 31)
)


class Generator(nn.Module):
    """ This is the style transform transformer module """

    def __init__(self , num_classes=1200,encode_one_hot = True):
        super().__init__()

        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.patch_embed = PatchEmbed(img_size=128, patch_size=4, in_chans=3, embed_dim=512)
        self.transformer = transformer.Transformer()
        # self.decode = decoder
        self.encode_one_hot = encode_one_hot

        self.deconv4 = ResidualBlockUp(512, 256, upsample=2)
        self.in4_d = nn.InstanceNorm2d(256, affine=True)

        self.deconv3 = ResidualBlockUp(256, 128, upsample=2)
        self.in3_d = nn.InstanceNorm2d(128, affine=True)

        self.deconv2 = ResidualBlockUp(128, 64, upsample=1)
        self.in2_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = ResidualBlockUp(64, 32, upsample=1)
        self.in1_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv0 = ResidualBlockUp(32, 16, upsample=1)
        self.in0_d = nn.InstanceNorm2d(16, affine=True)

        self.conv_end = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1), )

        # 在 64×64, C=256 上做一次自注意力（主要对高层语义和结构）
        self.attn4 = SelfAttention2d(256)
        # 在 128×128, C=128 上再做一次（可先注释掉，感觉稳定后再打开）
        self.attn3 = SelfAttention2d(128)

        self.flag_onehot = encode_one_hot

        self.embed = nn.Sequential(
            ConvLayer(1024, 512, kernel_size=3, stride=1,padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.averagepooling = nn.AdaptiveAvgPool2d(18)
        self.new_ps = nn.Conv2d(512, 512, (1, 1))

        if encode_one_hot:
            self.encode_one_hot = nn.Sequential(
                nn.Linear(num_classes, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),
            )
            self.encode_noise = nn.Sequential(
                ConvLayer(32, 64, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(64, affine=True),
                nn.Upsample(scale_factor=2),

                ConvLayer(64, 128, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(128, affine=True),
                nn.Upsample(scale_factor=2),

                ConvLayer(128, 256, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(256, affine=True),
                nn.Upsample(scale_factor=2),

                ConvLayer(256, 512, kernel_size=3, stride=1,padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(512, affine=True),
            )
        else:
            self.encode_one_hot = None


    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor,onehot=None):

        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(
                samples_c)  # support different-sized images padding is used for mask [tensor, mask]
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        ### Linear projection
        style = self.patch_embed(samples_s.tensors)  # 将风格图像嵌入为 Transformer 输入
        content = self.patch_embed(samples_c.tensors)

        # postional embedding is calculated in transformer.py
        content_pool = self.averagepooling(content)
        pos_c = self.new_ps(content_pool)
        pos_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])
        pos_s = None

        mask = None
        if onehot is not None and self.flag_onehot:
            noise = self.encode_one_hot(onehot)
            noise = noise.view(-1, 32, 4, 4)
            noise = self.encode_noise(noise)
            content = torch.cat((content, noise), 1)
            content = self.embed( content )

        hs = self.transformer(style, mask, content, pos_c, pos_s)  # 使用 Transformer 融合内容和风格特征。

        # 64×64, C=256
        out = self.deconv4(hs)
        out = self.in4_d(out)
        out = self.attn4(out)  # ★ 自注意力 1

        # out = self.deconv3(out)
        # out = self.in3_d(out)
        # out = self.attn3(out)
        # out = self.in4_d(self.deconv4(hs))  # (B, 256, 64, 64)
        out = self.in3_d(self.deconv3(out))  # (B, 128, 128, 128)
        out = self.in2_d(self.deconv2(out))  # (B, 64, 128, 128)
        out = self.in1_d(self.deconv1(out))  # (B, 32, 128, 128)
        out = self.in0_d(self.deconv0(out))  # (B, 16, 128, 128)
        # out = self.deconv1(out)  # (B, 32, 128, 128)
        # out = self.deconv0(out) # (B, 16, 128, 128)
        Ics = self.conv_end(out)  # (B, 3, 128, 128)

        return Ics


class Discriminator(nn.Module):
    def __init__(self, input_nc=6, num_classes=1200, img_size=128, **kwargs):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.conv1 = ResidualBlockDown(input_nc, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.conv4 = ResidualBlockDown(256, 512)
        if img_size==128:
            self.conv5 = ResidualBlockDown(512, 512)

        self.dense0 = nn.Linear(8192, 1024)
        self.dense1 = nn.Linear(1024, 1)

    def forward(self, x, high_res=0):
        out = x  # [B, 6, 64, 64]
        # Encode
        out_0 = (self.conv1(out))  # [B, 64, 32, 32]
        out_1 = (self.conv2(out_0))  # [B, 128, 16, 16]
        out_3 = (self.conv3(out_1))  # [B, 256, 8, 8]
        out = (self.conv4(out_3))  # [B, 512, 4, 4]
        if self.img_size==128:
            out = (self.conv5(out))  # [B, 512, 4, 4]

        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense0(out), 0.2, inplace=True)
        out = F.leaky_relu(self.dense1(out), 0.2, inplace=True)
        return out

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=5, num_classes=1200,**kwargs):
        """
        input_nc: 输入通道数，通常为图像 + 引导信息，如 [图像, 关键点] 拼接后的通道
        ndf: 初始卷积通道数
        n_layers: 卷积层数，4 是经典配置
        """
        super(PatchGANDiscriminator, self).__init__()
        layers = []

        # 第 1 层：不使用归一化
        layers += [
            nn_utils.spectral_norm( nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # 最后一层卷积：stride=1 保证输出 spatial map
        layers += [
            nn_utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )# 输出 shape: [B, 1, H/16, W/16]

        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn_utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            # sequence += [
            #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
            #               kernel_size=kw, stride=2, padding=padw, bias=False),
            #     nn_utils.spectral_norm((ndf * nf_mult)),
            #     nn.LeakyReLU(0.2, True)
            # ]
            conv = spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False)
            )
            sequence += [
                conv,
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        #               kernel_size=kw, stride=1, padding=padw, bias=False),
        #     nn_utils.spectral_norm((ndf * nf_mult)),
        #     nn.LeakyReLU(0.2, True)
        # ]
        conv = spectral_norm(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False)
        )
        sequence += [
            conv,
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x, return_features=False):
        features = []
        out = x
        for layer in self.model:
            out = layer(out)
            features.append(out)
        if return_features:
            return features
        else:
            return [out]

# ------------------------------
#   Multi-Scale 判别器
# ------------------------------
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, num_D=3,num_classes=None, encode_one_hot=False):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(input_nc, ndf, n_layers)
            for _ in range(num_D)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x, return_features=False):
        results = []
        input_downsampled = x
        for D in self.discriminators:
            out = D(input_downsampled, return_features=return_features)
            results.append(out)
            input_downsampled = self.downsample(input_downsampled)
        return results




# region Residual Blocks 通过卷积和平均池化（Average Pooling）实现下采样
class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual) #卷积匹配通道数，增强特征表达
        residual = F.avg_pool2d(residual, 2)#池化确保残差路径与主路径的分辨率一致，避免维度不匹配

        # Merge
        out = residual + out
        return out

#通过上采样（Upsampling）和卷积恢复分辨率
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        # General
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True) #实例归一化，对每个样本（实例）的特征进行归一化。
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = self.norm_r1(x)#归一化特征，确保每层的输入分布一致，稳定训练。
        out = F.relu(out)
        out = self.upsample(out) #上采样会改变特征图的分辨率，可能导致特征分布不稳定。
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual) #残差路径使用 1x1 卷积（conv_l）调整通道数

        # Merge
        out = residual + out
        return out

# class Blur(nn.Module):
#     def __init__(self, kernel=(1, 2, 1), pad=1):
#         super().__init__()
#         k = torch.tensor(kernel, dtype=torch.float32)
#         k = (k[:, None] * k[None, :])  # 3x3
#         k = k / k.sum()
#         self.register_buffer('kernel', k[None, None, :, :])
#         self.pad = nn.ReflectionPad2d(pad)
#
#     def forward(self, x):
#         c = x.shape[1]
#         k = self.kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
#         x = self.pad(x)                              # pad=1
#         return F.conv2d(x, k, groups=c)              # 3x3 → 尺寸不变
#
#
# class ResidualBlockUp(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2,
#                  use_in_first=True, use_in_second=False, res_scale=0.2, use_blur=True):
#         super().__init__()
#         self.res_scale = res_scale
#         self.scale = upsample
#         self.use_blur = use_blur
#         self.blur = Blur() if use_blur else nn.Identity()
#
#         # 右支
#         self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True) if use_in_first else nn.Identity()
#         self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)   # 3×3 + pad=1（保持尺寸）
#         self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True) if use_in_second else nn.Identity()
#         self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)  # 3×3 + pad=1
#
#         # 左支
#         self.conv_l  = ConvLayer(in_channels, out_channels, 1, 1, padding=0)       # 1×1 + pad=0（保持尺寸）
#
#     def _upsample(self, x):
#         if self.scale == 1:
#             return x
#         try:
#             return F.interpolate(x, scale_factor=self.scale, mode='bilinear',
#                                  align_corners=False, antialias=True)
#         except TypeError:
#             return F.interpolate(x, scale_factor=self.scale, mode='bilinear',
#                                  align_corners=False)
#
#     def forward(self, x):
#         # 右支
#         out = self.norm_r1(x)
#         out = F.relu(out, inplace=True)
#         out = self._upsample(out)
#         out = self.blur(out)
#         out = self.conv_r1(out)
#
#         out = self.norm_r2(out)
#         out = F.relu(out, inplace=True)
#         out = self.conv_r2(out)
#
#         # 左支
#         residual = self._upsample(x)
#         residual = self.blur(residual)
#         residual = self.conv_l(residual)
#
#         out = residual + self.res_scale * out
#         # 防御式断言（调试期可保留几次）
#         assert out.shape[-2:] == residual.shape[-2:], f"{out.shape} vs {residual.shape}"
#         return out

#在编码器和解码器之间，增强低分辨率特征的表达能力。
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out

#基础卷积层，封装了带反射填充（Reflection Padding）和谱归一化（Spectral Normalization）的卷积操作。
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
# endregion