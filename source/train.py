#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""
import contextlib
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import numpy as np
import importlib
import random
import math
from sacred import Experiment
import util_func as util_func
import util_loss as util_loss
import util_data as util_data
import sys
import os
import torch.distributed as dist
import matplotlib.pyplot as plt
from util_loss import VGGPerceptualLoss,ColorConsistencyLoss


sys.path.append(os.path.join(os.path.dirname(__file__), 'arch'))

structformer_exp = Experiment()
loss_history = {"critic": [], "gen_adv": [], "gen_l2": [], "siam": []}
grad_history = {"critic": [], "generator": [], "siamese": []}
@structformer_exp.config
def my_config():
    TRAIN_PARAMS = {
        'ARCH_NUM': 'unet_flex',  #生成器的网络架构
        'ARCH_SIAM': 'resnet_siam', #指定 Siamese 网络（身份判别器）的架构。
        'FILTER_NUM': 32, #卷积层的基础通道数
        'LEARNING_RATE': 0.00007,
        'LEARNING_RATE_G': 0.00004,
        'LEARNING_RATE_C': 0.00001,
        'LEARNING_RATE_S': 0.00007,
        'MIN_LEARNING_RATE': 1e-6,
        'LR_DECAY': 5e-5,  # 新增：学习率衰减系数（相当于原来的 args.lr_decay）
        'WARMUP_ITER': 1000,  # warm-up阶段的迭代次数
        'WARMUP_ITER_G': 6000,  # 生成器 - 5%
        'WARMUP_ITER_C': 6000,  # 鉴别器 - 5%0
        'WARMUP_ITER_S': 1000,  # Siamese - 7.5%
        'WARMUP_START_FRAC_C': 0.1,
        'FLAG_GPU': True,
        'EPOCHS_NUM': 50,
        'EPOCH_START': 0,

        'RESUME': False,
        # 可以是完整路径，也可以只是文件名（会自动拼到 model_dir 下）
        'CKPT_PATH': '/home/user/project/structformer/models-2/ciagan_Aunet_flex_Dceleba_Tcheck/Aunet_flex_Dceleba_Tcheck_last.ckpt',

        'ITER_CRITIC': 1,#判别器的迭代次数
        'ITER_GENERATOR': 1,#生成器的迭代次数
        'ITER_SIAMESE':1,#Siamese网络的迭代次数-

        'GAN_TYPE':'lsgan',#wgangp lsgan hinge
        'FLAG_SIAM_MASK': False, #是否在 Siamese 网络中使用蒙板（mask）0
    }

    DATA_PARAMS = {
        'DATA_PATH': '/home/user/project/structformer/',  #数据集存放路径
        'DATA_SET': 'celeba_new',    #数据集名称
        'LABEL_NUM': 1200,       #标签数量或样本数量
        'WORKERS_NUM': 0,        #数据加载的工作进程数
        'BATCH_SIZE': 8,        #每批次样本数量
        'IMG_SIZE': 128,         #图像尺寸
        'FLAG_DATA_AUGM': True,  #是否启用数据增强
    }

    OUTPUT_PARAMS = {
        'RESULT_PATH': '/home/user/project/structformer/results/', #结果保存路径
        'MODEL_PATH': '/home/user/project/structformer/models-2/',   #权重保存历经
        'LOG_ITER': 50, #每隔多少次迭代记录一次日志
        'SAVE_EPOCH': 5, #每隔5个epoch保存一次模型
        'SAVE_CHECKPOINT': 50,
        'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'main',
        'SAVE_IMAGES': True,
        'PROJECT_NAME': 'ciagan',
        'EXP_TRY': 'check',
        'COMMENT': "Default",

    }

# sacred library
load_data = structformer_exp.capture(util_data.load_data, prefix='DATA_PARAMS')
load_model = structformer_exp.capture(util_func.load_model)
set_comp_device = structformer_exp.capture(util_func.set_comp_device, prefix='TRAIN_PARAMS')
set_output_folders = structformer_exp.capture(util_func.set_output_folders)
set_model_name = structformer_exp.capture(util_func.set_model_name)

def _get_base_lr(optimizer, train_params, which=None):
    """
    返回用于调度的 base_lr（优先级：optimizer.initial_lr -> TRAIN_PARAMS['LEARNING_RATE_{which}']
    -> TRAIN_PARAMS['LEARNING_RATE']）。如果都没有，返回一个小的安全值并打印警告。
    """
    # 优先读取 optimizer 保存的 initial_lr（如果你在初始化后写入了）
    pg0 = optimizer.param_groups[0]
    base_lr = pg0.get('initial_lr', None)
    if base_lr is None:
        # 尝试从 train_params 中读取 per-optimizer 或全局 LEARNING_RATE
        if which is not None:
            key = f'LEARNING_RATE_{which}'
            base_lr = train_params.get(key, None)
        if base_lr is None:
            base_lr = train_params.get('LEARNING_RATE', None)

    if base_lr is None:
        base_lr = 1e-6
        print("[WARN] base_lr not found in optimizer or train_params; fallback to 1e-6")
    return float(base_lr)


def update_learning_rate(optimizer, global_iter, train_params, total_training_steps, which=None):
    """
    which: optional 'G'/'C'/'S' to prefer per-optimizer train_params key
    Returns the updated lr (float).
    """
    warmup_key = f'WARMUP_ITER_{which}' if which else 'WARMUP_ITER'
    warmup_iter = int(train_params.get(warmup_key, 0))
    base_lr = _get_base_lr(optimizer, train_params, which=which)
    lr_min = float(train_params.get('MIN_LEARNING_RATE', 1e-6))

    # 允许为不同优化器设置不同的 warmup 起始比例（例如 C=0.1、G=0.0）
    start_frac_key = f'WARMUP_START_FRAC_{which}' if which else 'WARMUP_START_FRAC'
    start_frac = float(train_params.get(start_frac_key, 0.0))  # 默认保持你原先的行为=0

    if warmup_iter > 0 and global_iter < warmup_iter:
        lr = warmup_learning_rate(
            optimizer, global_iter, base_lr, warmup_iter,
            lr_min=lr_min, start_frac=start_frac
        )
    else:
        lr = adjust_learning_rate(
            optimizer, global_iter, base_lr, lr_min,
            total_training_steps, warmup_iter
        )
    return lr


def warmup_learning_rate(optimizer, global_iter, base_lr, warmup_iter, lr_min=1e-6, start_frac=0.0):
    """线性 warmup: 从 start_lr -> base_lr，其中 start_lr>=lr_min"""
    base_lr = float(base_lr)
    lr_min = float(lr_min)
    start_lr = max(lr_min, base_lr * float(start_frac))

    if warmup_iter <= 0:
        current_lr = base_lr
    else:
        progress = min(max(float(global_iter) / float(warmup_iter), 0.0), 1.0)
        current_lr = start_lr + (base_lr - start_lr) * progress

    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return current_lr



def adjust_learning_rate(optimizer, global_iter, base_lr, lr_min, total_training_steps, warmup_iter):
    """Cosine annealing using base_lr"""
    # 防御：确保 base_lr 和 lr_min 合理
    base_lr = float(base_lr)
    lr_min = float(lr_min)

    if total_training_steps is None:
        raise ValueError("需要 total_training_steps 来计算余弦退火")

    denom = max(1, (total_training_steps - warmup_iter))
    progress = float(global_iter - warmup_iter) / denom
    progress = min(max(progress, 0.0), 1.0)
    lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    # debug 打印（如需打开）
    # print(f"[DEBUG adjust] base_lr={base_lr}, progress={progress:.6f}, lr={lr:.8f}")
    return float(lr)

def get_grad_norm(model, norm_type=2):
    """计算模型参数梯度的 norm"""
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def edge_aware_loss(output_gen, im_faces, im_msk):

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=output_gen.device)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=output_gen.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    def get_edge(img):
        """计算每张图像的梯度幅值（edge map）"""
        b, c, h, w = img.shape
        gx = F.conv2d(img.view(b * c, 1, h, w), sobel_x, padding=1)
        gy = F.conv2d(img.view(b * c, 1, h, w), sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        edge = edge.view(b, c, h, w)
        return edge.mean(dim=1, keepdim=True)  # 对RGB通道取平均 → 单通道边缘强度图

    # --- 2. 计算生成图与真实图的边缘 ---
    edge_fake = get_edge(output_gen)
    edge_real = get_edge(im_faces)

    # --- 3. 提取 mask 的边界区域 ---
    # 膨胀（背景扩张）
    dilated = F.max_pool2d(im_msk, kernel_size=3, stride=1, padding=1)
    # 腐蚀（背景收缩）
    eroded = -F.max_pool2d(-im_msk, kernel_size=3, stride=1, padding=1)
    # 边界带
    edge_mask = (dilated - eroded).abs()

    # --- 4. 在边界区域计算生成边缘与真实边缘的差异 ---
    weighted_diff = edge_mask * (edge_fake - edge_real).abs()

    # --- 5. 用边界区域像素数归一化 ---
    edge_pixel_count = edge_mask.sum() + 1e-8
    loss = weighted_diff.sum() / edge_pixel_count

    return loss


def _last_tensor(obj):
    """递归取结构中的最后一个 tensor（用于多尺度/多层输出）"""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        # 从后往前找第一个 tensor
        for x in reversed(obj):
            t = _last_tensor(x)
            if t is not None:
                return t
    return None  # 未找到

# =========================
# 对称增强 (DiffAugment 风格)
# 对 real 和 fake 都用同样的增强操作
# =========================
def random_flip(x):
    # x: [B, C, H, W]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])  # 水平翻转
    return x

def random_brightness(x, max_delta=0.1):
    # 只对 RGB 通道做亮度扰动，假设前3通道是图片
    if torch.rand(1).item() < 0.5:
        delta = (torch.rand(1).item() * 2 - 1) * max_delta  # [-max_delta, max_delta]
        x_rgb = x[:, :3] + delta
        x[:, :3] = x_rgb.clamp(0.0, 1.0)
    return x

def random_contrast(x, lower=0.9, upper=1.1):
    # 只对 RGB 通道做对比度变化
    if torch.rand(1).item() < 0.5:
        x_rgb = x[:, :3]
        mean = x_rgb.mean(dim=(2, 3), keepdim=True)
        factor = torch.empty(1).uniform_(lower, upper).item()
        x_rgb = (x_rgb - mean) * factor + mean
        x[:, :3] = x_rgb.clamp(0.0, 1.0)
    return x

def diff_augment(x):
    """
    对判别器输入做的一组轻量增强：
    - 水平翻转
    - 轻微亮度/对比度扰动
    注：real 和 fake 都要用这个函数
    """
    x = random_flip(x)
    x = random_brightness(x)
    x = random_contrast(x)
    return x


class Train_GAN():
    def __init__(self, model_info, device_comp, margin_contrastive=3, num_classes = 1200, gan_type = 'hinge'):
        self.model_info = model_info #保存模型信息
        self.device_comp = device_comp #保存计算设备
        self.num_classes = num_classes
        self.gan_type = gan_type
        self.criterion_gan = util_loss.GANLoss(gan_type).to(self.device_comp)
        #初始化 GAN 损失函数，并移动到指定设备。
        self.criterion_siamese = util_loss.ContrastiveLoss(margin_contrastive).to(self.device_comp)
        #初始化对比损失（Contrastive Loss），用于 Siamese 网络的训练。

        self.perceptual_loss = VGGPerceptualLoss(device=self.device_comp).to(self.device_comp)
        self.color_loss = ColorConsistencyLoss(kernel_size=21).to(self.device_comp)
        self.global_step = 0
        # self.r1_every = 16  # 每16步做一次
        # self.r1_gamma = 10.0  # 常用10
        # self.r1_step = 0

        project_root = os.path.dirname(os.path.dirname(__file__))  # /home/user/project/ciagan-master
        arcface_weight_path = os.path.join(
            project_root,
            'pretrained',
            'arcface-iresnet50',
            'arcface_mobilefacenet.pth'   # 你的权重名称
        )
        print("[DEBUG] arcface_weight_path =", arcface_weight_path)

        id_backbone = util_loss.build_arcface_backbone(
            arcface_weight_path,
            device=self.device_comp
        )

        self.id_loss_module = util_loss.IDLoss(
            id_backbone=id_backbone,
            lambda_pull=0.0,  # fake → target 身份
            lambda_push=0.5,  # 初始先不 push source，效果稳定后再考虑
            margin=0.3
        ).to(self.device_comp)

    def save_model(self, epoch_iter=0, mode_save=0):

        if mode_save == 0:
            torch.save(self.model_info['generator'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_G.pth')
            #保存生成器的权重。
            torch.save(self.model_info['critic'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_C.pth')
            torch.save(self.model_info['siamese'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_S.pth')

        elif mode_save == 1:
            torch.save(self.model_info['generator'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'G.pth')
            torch.save(self.model_info['critic'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'C.pth')
            torch.save(self.model_info['siamese'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'S.pth')

    def save_images(self, out, gt, target, inp, epoch_iter=0):
        viz_out_img = torch.clamp(out, 0., 1.)
        utils.save_image(viz_out_img, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_est.png")
        utils.save_image(gt, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_gt.png")
        utils.save_image(target, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + "_tar.png")
        utils.save_image(inp, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_inp.png")

    def _unwrap(self, m):
        return m.module if hasattr(m, "module") else m

    def save_checkpoint(self, tag, epoch_iter, global_iter_G, global_iter_C, global_iter_S):
        """保存完整训练快照（模型+EMA+优化器+步数）"""
        G = self._unwrap(self.model_info["generator"])
        C = self._unwrap(self.model_info["critic"])
        S = self._unwrap(self.model_info["siamese"])

        ckpt = {
            "version": 1,
            "tag": tag,
            "epoch": epoch_iter + 1,
            "global_iter_G": global_iter_G,
            "global_iter_C": global_iter_C,
            "global_iter_S": global_iter_S,
            "model": {
                "G": G.state_dict(),
                "C": C.state_dict(),
                "S": S.state_dict(),
                # 你的 EMA.shadow 本身就是完整 state_dict，直接存
                # "G_EMA": ({k: v.detach().cpu() for k, v in self.ema.shadow.items()}
                #           if hasattr(self, "ema") else None),
            },
            "optim": {
                "G": self.optimizer_G.state_dict() if hasattr(self, "optimizer_G") else None,
                "C": self.optimizer_C.state_dict() if hasattr(self, "optimizer_C") else None,
                "S": self.optimizer_S.state_dict() if hasattr(self, "optimizer_S") else None,
            },
            "train_params": {},  # 需要的话可把 TRAIN_PARAMS/OUTPUT_PARAMS 字段塞进来
        }

        os.makedirs(self.model_info["model_dir"], exist_ok=True)
        path = os.path.join(self.model_info["model_dir"], f"{self.model_info['model_name']}_{tag}.ckpt")
        torch.save(ckpt, path)
        print(f"[CKPT] Saved -> {path}")
        return path

    def reinit_loss(self):
        return [0] * 6, 0
    #处理一个批次的数据，准备训练所需的张量。im_faces[0]：原始图像。im_faces[1]：目标身份的参考图像，用于 Siamese 网络。
    def process_batch_data(self, input_batch, flag_same = False):
        im_faces, im_lndm, im_msk, im_ind = input_batch
        #将人脸图像转换为浮点型并移动到指定设备
        im_faces = [item.float().to(self.device_comp) for item in im_faces]
        im_lndm = [item.float().to(self.device_comp) for item in im_lndm]
        im_msk = [item.float().to(self.device_comp) for item in im_msk]

        #创建一个全 0 矩阵，形状为 [B, 1200]
        labels_one_hot = np.zeros((int(im_faces[0].shape[0]), self.num_classes))
        if len(im_ind)>1:
            labels_one_hot[np.arange(int(im_faces[1].shape[0])), im_ind[1]] = 1
        labels_one_hot = torch.tensor(labels_one_hot).float().to(self.device_comp)

        #两张图像是否属于同一身份。 与自身比较，label_same 是一个全 1 的张量（例如 [1, 1, ..., 1]）
        if flag_same:
            if len(im_ind) > 1:
                label_same = (im_ind[0]==im_ind[1])/1
            else:
                label_same = (im_ind[0]==im_ind[0])/1
            return im_faces, im_lndm, im_msk, labels_one_hot, label_same.to(self.device_comp)

        return im_faces, im_lndm, im_msk, labels_one_hot

    # Prepare inputs 拼接关键点和蒙版图像
    def input_train(self, im_faces, im_lndm, im_msk):
        input_repr = im_lndm #[B, 3, 128, 128]
        #im_msk[0]通常是 [B, 1, H, W] 值为1是蒙住。+
        #蒙板的作用：保留区域被清零（黑色），需要匿名化的区域保留原始值（但会被生成图像替换）
        # input_gen = torch.cat((input_repr, im_faces[0] * (1 - im_msk[0])), 1) #[B, 6, 128, 128]
        input_gen = im_faces[0] * (1 - im_msk[0])
        return input_gen, input_repr, im_msk

    def _id_preprocess(self,img):
        x = F.interpolate(img, size=(112, 112), mode='bilinear', align_corners=False)
        x = TF.gaussian_blur(x, kernel_size=3)  # 轻微模糊
        return x

    #训练 Siamese 网络：
    def train_siamese(self, num_iter_siamese=1, data_batch=None):
        loss_sum = 0
        if data_batch is None:
            # 这里的 next(self.data_iter) 理论上应该不会被执行，因为它会导致你之前的 StopIteration 错误。
            sample_batch = next(self.data_iter)
        else:
            sample_batch = data_batch  # <--- 【重要改动2】使用传入的 data_batch

        for p in self.model_info['siamese'].parameters():
            p.requires_grad = True #确保 Siamese 网络的参数可以更新

        for j in range(num_iter_siamese):
            # sample_batch = next(self.data_iter)
            im_faces, im_lndm, im_msk, im_onehot, label_data = self.process_batch_data(sample_batch, flag_same = True)
            #label_data：是否同一身份的标签（[B]，值为 0 或 1）
            self.optimizer_S.zero_grad()
            fc_real1 = self.model_info['siamese'](im_faces[0] * (im_msk[0] if self.flag_siam_mask else 1))
            fc_real2 = self.model_info['siamese'](im_faces[1] * (im_msk[1] if self.flag_siam_mask else 1))
            loss_S = self.criterion_siamese(fc_real1, fc_real2, label_data)

            loss_S.backward() #反向传播，计算梯度
            self.optimizer_S.step() #更新 Siamese 网络的参数
            loss_sum += loss_S.item()
        return loss_sum  #返回 Siamese 网络的对比损失

    def train_critic(self, num_iter_critic=1, data_batch=None):
        loss_sum = [0, 0]
        for p in self.model_info['critic'].parameters():
            p.requires_grad = True

        for j in range(num_iter_critic):
            # train region
            self.optimizer_C.zero_grad()

            # generate images from a frozen generator
            with torch.no_grad():
                if data_batch is None:
                    sample_batch = next(self.data_iter)
                else:
                    sample_batch = data_batch
                im_faces, im_lndm, im_msk, im_onehot = self.process_batch_data(sample_batch)
                input_gen, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)
                im_gen = self.model_info['generator'](input_gen, input_repr, onehot=im_onehot)


            # 计算判别器的损失，让它尽可能把 face_landmark_fake 识别为 "假"
            # loss_C_fake = self.criterion_gan(self.model_info['critic'](face_landmark_fake), False)
            # loss_C_real = self.criterion_gan(self.model_info['critic'](face_landmark_real), True)

            # loss_C_fake = self.criterion_gan(self.model_info['critic'](face_landmark_fake.detach(), False), False)
            # face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            # pred_fake = self.model_info['critic'](face_landmark_fake.detach())
            # loss_C_fake = self.criterion_gan(pred_fake, False)


            # output_gen = im_faces[0] * (1 - im_msk[0]) + im_gen * im_msk[0]
            # face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            # # loss_C_fake = self.criterion_gan(self.model_info['critic'](face_landmark_fake), False)
            # loss_C_fake = self.criterion_gan(self.model_info['critic'](face_landmark_fake.detach(), False), False)
            # # process real batch data
            # # sample_batch = next(self.data_iter)
            # im_faces, im_lndm, im_msk, im_onehot = self.process_batch_data(sample_batch)
            # output_real = im_faces[0]
            # _, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)
            #
            # # pred_real = self.model_info['critic'](face_landmark_real)
            # # loss_C_real = self.criterion_gan(pred_real, True)
            # face_landmark_real = torch.cat((output_real, input_repr), 1)
            # # loss_C_real = self.criterion_gan(self.model_info['critic'](face_landmark_real), True)
            # loss_C_real = self.criterion_gan(self.model_info['critic'](face_landmark_real, True), True)
            # # update
            # loss_D = loss_C_fake + loss_C_real
            output_gen = im_faces[0] * (1 - im_msk[0]) + im_gen * im_msk[0]

            # ---- fake: 生成图 + 条件，先拼，再做对称增强 ----
            face_landmark_fake = torch.cat((output_gen, input_repr), 1)  # [B, 3+3, H, W]
            face_landmark_fake_aug = diff_augment(face_landmark_fake.detach())
            loss_C_fake = self.criterion_gan(
                self.model_info['critic'](face_landmark_fake_aug, False),
                False
            )

            # ---- real: 真图 + 条件，同样的增强 ----
            im_faces, im_lndm, im_msk, im_onehot = self.process_batch_data(sample_batch)
            output_real = im_faces[0]
            _, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)

            face_landmark_real = torch.cat((output_real, input_repr), 1)
            face_landmark_real_aug = diff_augment(face_landmark_real)
            loss_C_real = self.criterion_gan(
                self.model_info['critic'](face_landmark_real_aug, True),
                True
            )

            loss_D = loss_C_fake + loss_C_real

            if self.gan_type == 'wgangp':
                grad_penalty = util_loss.cal_gradient_penalty(self.model_info['critic'], face_landmark_real,
                                                              face_landmark_fake.detach(), self.device_comp)
                loss_D += grad_penalty

            # if self.gan_type in ('hinge', 'lsgan'):
            #     self.r1_step += 1
            #     if (self.r1_step % self.r1_every) == 0:
            #         real_in = face_landmark_real.detach().requires_grad_(True)
            #         real_out = self.model_info['critic'](real_in, True)  # 你的 D(x, True) 返回可能是 tensor 或 list
            #
            #         # 递归把“最终 logits”求和成一个标量
            #         def _sum_final_logits(o):
            #             if isinstance(o, (list, tuple)):
            #                 s = 0
            #                 for x in o:
            #                     # 若某个尺度是 [feat1, feat2, logits] 这样的结构，取最后一个当 logits
            #                     if isinstance(x, (list, tuple)):
            #                         s = s + _sum_final_logits(x[-1])
            #                     else:
            #                         s = s + _sum_final_logits(x)
            #                 return s
            #             # tensor
            #             return o.sum()
            #
            #         real_sum = _sum_final_logits(real_out)
            #
            #         grad_real = torch.autograd.grad(
            #             outputs=real_sum, inputs=real_in, create_graph=True
            #         )[0]
            #
            #         # 归一化版 R1：用 mean，避免总损失出现 1e5 的巨大数值
            #         r1 = 0.5 * self.r1_gamma * grad_real.pow(2).mean()
            #         loss_D = loss_D + r1

            loss_D.backward()
            self.optimizer_C.step()  # 更新判别器的参数。

            # for logging and visualization
            loss_sum[0] += loss_D.item()  # 累加总判别器损失。
            loss_sum[1] += loss_C_real.item()  # 累加真实图像的判别损失。

        return loss_sum

    def train_generator(self, num_iter_generator=1, flag_siamese=False, data_batch=None):
        loss_sum = [0, 0]

        # freeze D/S, unfreeze G
        for p in self.model_info['critic'].parameters():
            p.requires_grad = False
        for p in self.model_info['siamese'].parameters():
            p.requires_grad = False
        for p in self.model_info['generator'].parameters():
            p.requires_grad = True

        im_faces, im_lndm, output_gen = [], [], []
        for _ in range(num_iter_generator):
            sample_batch = next(self.data_iter) if data_batch is None else data_batch
            im_faces, im_lndm, im_msk, im_onehot, label_same = self.process_batch_data(sample_batch, flag_same=True)

            # ===== G forward =====
            self.optimizer_G.zero_grad()
            input_gen, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)
            im_gen = self.model_info['generator'](input_gen, input_repr, onehot=im_onehot)
            output_gen = im_faces[0] * (1 - im_msk[0]) + im_gen * im_msk[0]

            #deid_loss
            with torch.no_grad():
                img_src_for_id = self._id_preprocess(im_faces[0])  # 原图
            img_fake_for_id = self._id_preprocess(output_gen)  # 生成图（需要梯度）
            feat_src = self.id_loss_module.id_backbone(img_src_for_id)
            feat_fake = self.id_loss_module.id_backbone(img_fake_for_id)

            cos_sim = F.cosine_similarity(feat_fake, feat_src, dim=1)  # [B]
            # de-ID loss：希望 cos_sim 越小越好 → 取平均
            loss_id = cos_sim.mean()
            lambda_id  = 0.5  # 先用一个比较小的权重


            # face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            face_landmark_fake_aug = diff_augment(face_landmark_fake)

            x_src = im_faces[0]  # 原图
            x_tgt = im_faces[1] if len(im_faces) > 1 else None
            PP = 0.8

            loss_perceptual = self.perceptual_loss(
                x_fake=im_gen,  # 只在被替换区域里对齐
                x_src=x_src,
                x_tgt=x_tgt,
                mask=im_msk[0],
                pp=PP
            )
            # loss_G_rec = F.l1_loss(im_gen * im_msk[0], im_faces[1] * im_msk[0])
            # diff = (im_gen - im_faces[1]).abs() * im_msk[0]
            # den = im_msk[0].sum() * im_gen.shape[1] + 1e-8
            # loss_G_rec = diff.sum() / den
            loss_G_rec = F.l1_loss(im_gen * im_msk[0], im_faces[0] * im_msk[0])
            loss_color = self.color_loss(output_gen, im_faces[0], im_msk[0])
            # 定义损失权重
            lambda_gan = 1
            lambda_rec = 3
            lambda_edge = 1.5
            lambda_perc = 0.8
            lambda_fm = 0
            lambda_color = 0.0
            loss_G_siam = 0
            loss_edge = edge_aware_loss(output_gen, im_faces[0], im_msk[0])

            # 特征匹配损失
            # pred_real = self.model_info['critic'](face_landmark_real, return_features=True)
            pred_fake = self.model_info['critic'](face_landmark_fake_aug, return_features=True)
            #
            # loss_fm = 0
            # for D_real, D_fake in zip(pred_real, pred_fake):
            #     num_layers = len(D_real)
            #     selected_layers = range(1, min(4, num_layers - 1))
            #     for i in selected_layers:
            #         loss_fm += F.l1_loss(D_fake[i], D_real[i].detach())
            #
            # loss_fm = loss_fm / (len(pred_real) * len(selected_layers))

            # GAN 损失
            loss_G_gan = 0
            for pred in pred_fake:
                for i, p in enumerate(pred):
                    weight = 1.0 / (2 ** (len(pred) - i - 1))
                    loss_G_gan += weight * self.criterion_gan(p, True)

            loss_G_gan /= len(pred_fake)
            # loss_G_gan = self.criterion_gan(self.model_info['critic'](face_landmark_fake), True)
            # --------- ArcFace 身份损失（基础版本）---------
            # lambda_id = 0.2# 可以从 0.5 开始试，视效果调整

            # loss_id, id_stats = self.id_loss_module(
            #     img_src=im_faces[0],
            #     img_tgt=im_faces[1],
            #     img_fake=output_gen
            # )
            # loss_id=0
            # 总生成器损失（加上身份项）
            loss_G = (lambda_gan * loss_G_gan +
                      lambda_rec * loss_G_rec +
                      lambda_edge * loss_edge +
                      lambda_perc * loss_perceptual +
                      lambda_id * loss_id+
                      lambda_color * loss_color)

            loss_G.backward()
            self.optimizer_G.step()

            with torch.no_grad():
                w_adv = loss_G_gan.detach()
                w_rec = (lambda_rec * loss_G_rec).detach()
                w_edge = (lambda_edge * loss_edge).detach()
                w_perc = (lambda_perc * loss_perceptual).detach()
                # w_fm = (lambda_fm * loss_fm).detach()
                w_id = (lambda_id * loss_id).detach()
                # w_id=0
                w_color = (lambda_color * loss_color).detach()

                total_w = (w_adv + w_rec + w_edge + w_perc + w_id).clamp_min(1e-8)
                # ===== 每 LOG_ITER 步打印一次（按全局 G 步数计），且仅主进程打印 =====
                log_every = int(self.model_info.get('LOG_ITER', 50))
                # 当前这一次 forward/backward 完成后的“即将成为”的步号
                step_idx = int(self.model_info.get('global_iter_G', 0)) + 1

                is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
                if step_idx > 0 and (step_idx % log_every == 0) and is_main:
                    adv_pct = (w_adv / total_w * 100).item()
                    rec_pct = (w_rec / total_w * 100).item()
                    edge_pct = (w_edge / total_w * 100).item()
                    perc_pct = (w_perc / total_w * 100).item()
                    id_pct = (w_id / total_w * 100).item()
                    color_pct = (w_color / total_w * 100).item()
                    print(
                        f"[G breakdown @ step {step_idx}] total={total_w.item():.4f} | "
                        f"adv:{adv_pct:5.1f}% rec:{rec_pct:5.1f}% edge:{edge_pct:5.1f}% "
                        f"perc:{perc_pct:5.1f}% id:{id_pct:5.1f}% color:{color_pct:5.1f}%"
                    )
            # EMA（如有）
            # logging
            loss_sum[0] += loss_G_gan.item()
            # 下面是你原来的可视化上色逻辑，保留即可
            for l_iter in range(len(label_same)):
                if label_same[l_iter] == 1:
                    im_faces[1][l_iter, 1, :, :] += 0.2
                    im_faces[1][l_iter, 2, :, :] += 0.2
                else:
                    im_faces[1][l_iter, 0, :, :] += 0.2

        return loss_sum, im_faces, im_lndm, output_gen

    @structformer_exp.capture
    def train_model(self, loaders, TRAIN_PARAMS, OUTPUT_PARAMS):
        # === 关键：起始 epoch 从 model_info 里拿（如果没有就用配置里的） ===
        epoch_start = int(self.model_info.get('epoch_start',
                                              TRAIN_PARAMS.get('EPOCH_START', 0)))

        lr_g = TRAIN_PARAMS.get('LEARNING_RATE_G', TRAIN_PARAMS['LEARNING_RATE'])
        lr_c = TRAIN_PARAMS.get('LEARNING_RATE_C', TRAIN_PARAMS['LEARNING_RATE'])
        lr_s = TRAIN_PARAMS.get('LEARNING_RATE_S', TRAIN_PARAMS['LEARNING_RATE'])

        self.optimizer_G = optim.Adam(self.model_info['generator'].parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.optimizer_C = optim.Adam(self.model_info['critic'].parameters(), lr=lr_c, betas=(0.0, 0.9))
        self.optimizer_S = optim.Adam(self.model_info['siamese'].parameters(), lr=lr_s, betas=(0.0, 0.9))

        # 保存 initial_lr
        for opt in (self.optimizer_G, self.optimizer_C, self.optimizer_S):
            for pg in opt.param_groups:
                if 'initial_lr' not in pg:
                    pg['initial_lr'] = float(pg.get('lr', 0.0))

        print("Saved initial lr -> G:", self.optimizer_G.param_groups[0]['initial_lr'],
              "C:", self.optimizer_C.param_groups[0]['initial_lr'])

        # === 如果是从 checkpoint 恢复，这里把 optimizer 的状态也加载回来 ===
        if self.model_info.get('optim_G') is not None:
            self.optimizer_G.load_state_dict(self.model_info['optim_G'])
        if self.model_info.get('optim_C') is not None:
            self.optimizer_C.load_state_dict(self.model_info['optim_C'])
        if self.model_info.get('optim_S') is not None:
            self.optimizer_S.load_state_dict(self.model_info['optim_S'])

        self.flag_siam_mask = TRAIN_PARAMS['FLAG_SIAM_MASK']

        # --- 计算总步数 ---
        num_iter_siamese = TRAIN_PARAMS['ITER_SIAMESE']
        num_iter_critic = TRAIN_PARAMS['ITER_CRITIC']
        num_iter_generator = TRAIN_PARAMS['ITER_GENERATOR']

        steps_per_epoch = len(loaders[0])
        epochs_to_train = TRAIN_PARAMS['EPOCHS_NUM'] - epoch_start

        total_training_steps_G = epochs_to_train * steps_per_epoch * num_iter_generator
        total_training_steps_C = epochs_to_train * steps_per_epoch * num_iter_critic
        total_training_steps_S = epochs_to_train * steps_per_epoch * num_iter_siamese
        self.model_info['total_training_steps'] = total_training_steps_C
        print(f"Total actual training steps for LR scheduler: {total_training_steps_C}")

        self.model_info['LOG_ITER'] = OUTPUT_PARAMS.get('Loss_ITER', 50)

        # 从 checkpoint 里恢复的全局步数（如果没有就从 0 开始）
        global_iter_G = self.model_info.get('global_iter_G_start', 0)
        global_iter_C = self.model_info.get('global_iter_C_start', 0)
        global_iter_S = self.model_info.get('global_iter_S_start', 0)
        self.model_info['global_iter_G'] = global_iter_G

        # --- 训练循环 ---
        for e_iter in range(epoch_start, TRAIN_PARAMS['EPOCHS_NUM']):
            epoch_iter = e_iter
            loss_sum, iter_count = self.reinit_loss()
            self.data_iter = iter(loaders[0])

            for st_iter in range(steps_per_epoch):
                try:
                    sample_batch = next(self.data_iter)
                except StopIteration:
                    print(f"警告：在 epoch {epoch_iter} 的步数 {st_iter}/{steps_per_epoch} 时数据已耗尽。")
                    break

                # --- Critic ---
                for _ in range(num_iter_critic):
                    lr_C = update_learning_rate(self.optimizer_C, global_iter_C, TRAIN_PARAMS, total_training_steps_C)
                    loss_values_critic = self.train_critic(num_iter_critic=num_iter_critic, data_batch=sample_batch)
                    loss_sum[0] += loss_values_critic[0]
                    loss_sum[4] += loss_values_critic[1]
                    global_iter_C += 1

                # --- Generator ---
                for _ in range(num_iter_generator):
                    lr_G = update_learning_rate(self.optimizer_G, global_iter_G, TRAIN_PARAMS, total_training_steps_G)
                    train_out = self.train_generator(
                        num_iter_generator=num_iter_generator,
                        flag_siamese=False if num_iter_siamese == 0 else True,
                        data_batch=sample_batch
                    )
                    loss_values_gen, im_faces, im_lndm, im_gen = train_out[:4]
                    loss_sum[1] += loss_values_gen[0]
                    loss_sum[2] += loss_values_gen[1]
                    global_iter_G += 1
                    self.model_info['global_iter_G'] = global_iter_G

                # --- Siamese ---
                for _ in range(num_iter_siamese):
                    lr_S = update_learning_rate(self.optimizer_S, global_iter_S, TRAIN_PARAMS, total_training_steps_S)
                    self.train_siamese(num_iter_siamese=num_iter_siamese, data_batch=sample_batch)
                    global_iter_S += 1

                iter_count += 1

                # log
                if (st_iter + 1) % OUTPUT_PARAMS['LOG_ITER'] == 0:
                    print(self.model_info['model_name'],
                          'Epoch [{}/{}], Step [{}/{}], Loss C: {:.4f}, G: {:.4f}, S: {:.4f}'
                          .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'],
                                  st_iter + 1, self.model_info['total_steps'],
                                  loss_sum[0] / iter_count, loss_sum[1] / iter_count, loss_sum[3] / iter_count))

                    lr_C = update_learning_rate(self.optimizer_C, global_iter_C, TRAIN_PARAMS, total_training_steps_C,
                                                which='C')
                    lr_G = update_learning_rate(self.optimizer_G, global_iter_G, TRAIN_PARAMS, total_training_steps_G,
                                                which='G')
                    print(f"Current LR -> Generator: {lr_G:.6f}, Critic: {lr_C:.6f}")

                    grad_norm_c = get_grad_norm(self.model_info['critic'])
                    grad_norm_g = get_grad_norm(self.model_info['generator'])
                    grad_norm_s = get_grad_norm(self.model_info['siamese'])

                    grad_history["critic"].append(grad_norm_c)
                    grad_history["generator"].append(grad_norm_g)
                    grad_history["siamese"].append(grad_norm_s)

                    loss_history["critic"].append(loss_sum[0] / iter_count)
                    loss_history["gen_adv"].append(loss_sum[1] / iter_count)
                    loss_history["gen_l2"].append(loss_sum[2] / iter_count)
                    loss_history["siam"].append(loss_sum[3] / iter_count)

                    loss_sum, iter_count = self.reinit_loss()
                    print(f"[GradNorm] C: {grad_norm_c:.4f}, G: {grad_norm_g:.4f}, S: {grad_norm_s:.4f}")

            # 保存模型 & checkpoint
            if (epoch_iter + 1) % OUTPUT_PARAMS['SAVE_EPOCH'] == 0:
                self.save_model(mode_save=0)
                self.save_checkpoint(
                    tag="last",
                    epoch_iter=epoch_iter,
                    global_iter_G=global_iter_G,
                    global_iter_C=global_iter_C,
                    global_iter_S=global_iter_S
                )

                if (epoch_iter + 1) % OUTPUT_PARAMS['SAVE_CHECKPOINT'] == 0:
                    self.save_model(epoch_iter=epoch_iter, mode_save=1)
                    self.save_checkpoint(
                        tag=f"ep{epoch_iter + 1:05d}",
                        epoch_iter=epoch_iter,
                        global_iter_G=global_iter_G,
                        global_iter_C=global_iter_C,
                        global_iter_S=global_iter_S
                    )


@structformer_exp.automain
def run_exp(TRAIN_PARAMS):
    ##### INITIAL PREPARATIONS
    model_name = set_model_name()  # 设置模型名称。
    model_dir, res_dir = set_output_folders(model_name)  # 设置模型和结果的保存目录。
    device_comp = set_comp_device()  # 设置计算设备
    resume = TRAIN_PARAMS.get('RESUME', False)
    epoch_start = TRAIN_PARAMS.get('EPOCH_START', 0)
    ckpt = None
    global_iter_G_start = global_iter_C_start = global_iter_S_start = 0
    optim_G = optim_C = optim_S = None

    if resume:
        ckpt_path_cfg = TRAIN_PARAMS.get('CKPT_PATH', '')
        if ckpt_path_cfg:
            # 如果给的是相对文件名，就拼到 model_dir 下面
            if os.path.isabs(ckpt_path_cfg):
                ckpt_path = ckpt_path_cfg
            else:
                ckpt_path = os.path.join(model_dir, ckpt_path_cfg)
        else:
            # 没指定就默认用 "<model_name>_last.ckpt"
            ckpt_path = os.path.join(model_dir, f"{model_name}_last.ckpt")

        print(f"[CKPT] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device_comp)

        epoch_start = int(ckpt.get('epoch', 0))
        print(f"[CKPT] Resume from epoch {epoch_start}")

        global_iter_G_start = ckpt.get('global_iter_G', 0)
        global_iter_C_start = ckpt.get('global_iter_C', 0)
        global_iter_S_start = ckpt.get('global_iter_S', 0)

        optim_dict = ckpt.get('optim', {}) or {}
        optim_G = optim_dict.get('G', None)
        optim_C = optim_dict.get('C', None)
        optim_S = optim_dict.get('S', None)

    ##### PREPARING DATA
    loader_train, total_steps, label_num = load_data(mode_train=True)
    loaders = [loader_train]
    epoch_start_for_load_model = 0 if resume else epoch_start

    # ====== 这里插入一次性的 sampler 检查 ======
    DEBUG_SAMPLER = False
    if DEBUG_SAMPLER:
        dataset = loader_train.dataset
        sampler = loader_train.sampler

        it = iter(sampler)
        first_batch_pairs = [next(it) for _ in range(loader_train.batch_size)]

        print("Check first batch pairs (class of src vs tgt):")
        for b, (idx_src, idx_tgt) in enumerate(first_batch_pairs[:10]):  # 先看前 10 对就够了
            label_src = dataset.im_label[idx_src]
            label_tgt = dataset.im_label[idx_tgt]
            print(f"pair {b}: src_label={label_src}, tgt_label={label_tgt}")
            assert label_src != label_tgt, "发现同身份 pair，说明 sampler 逻辑还有问题！"
    # ====== 检查结束 ======


    ##### PREPARING MODELS
    generator = load_model(model_dir, model_name, 'Generator', device_comp, TRAIN_PARAMS['ARCH_NUM'],
                           epoch_start=epoch_start_for_load_model, label_num=label_num)
    critic = load_model(model_dir, model_name, 'MultiscaleDiscriminator', device_comp, TRAIN_PARAMS['ARCH_NUM'],
                        epoch_start=epoch_start_for_load_model)

    if TRAIN_PARAMS['ARCH_SIAM'][:6] == 'resnet':
        siamese = load_model(model_dir, model_name, 'ResNet', device_comp, TRAIN_PARAMS['ARCH_SIAM'],
                             epoch_start=epoch_start_for_load_model)
    elif TRAIN_PARAMS['ARCH_SIAM'][:4] == 'siam':
        siamese = load_model(model_dir, model_name, 'PatchGANDiscriminator', device_comp, TRAIN_PARAMS['ARCH_SIAM'],
                             epoch_start=epoch_start_for_load_model)

    if ckpt is not None:
        state_model = ckpt.get('model', {})
        if 'G' in state_model:
            generator.load_state_dict(state_model['G'])
        if 'C' in state_model:
            critic.load_state_dict(state_model['C'])
        if 'S' in state_model:
            siamese.load_state_dict(state_model['S'])
    ##### PASSING INFO
    model_info = {'generator': generator,
                  'critic': critic,
                  'siamese': siamese,
                  'model_dir': model_dir,
                  'model_name': model_name,
                  'res_dir': res_dir,
                  'total_steps': total_steps,
                  'device_comp': device_comp,
                  'label_num': label_num,
                  # === 关键：起始 epoch / 步数 / optimizer 状态 传进 trainer ===
                  'epoch_start': epoch_start,
                  'global_iter_G_start': global_iter_G_start,
                  'global_iter_C_start': global_iter_C_start,
                  'global_iter_S_start': global_iter_S_start,
                  'optim_G': optim_G,
                  'optim_C': optim_C,
                  'optim_S': optim_S,
                  }

    ##### INITIALIZE AND START TRAINING
    trainer = Train_GAN(model_info=model_info, device_comp=device_comp, num_classes=label_num,
                        gan_type=TRAIN_PARAMS['GAN_TYPE'])
    trainer.train_model(loaders=loaders)
