#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import os
from os import listdir, mkdir
from os.path import isfile, join, isdir, exists
import numpy as np
import importlib
import random
import math
from PIL import Image
from collections import defaultdict
import cv2
import numbers

# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, label_num=1200, transform_fnc=transforms.Compose([transforms.ToTensor()]),
#                  img_size = 128, flag_init=True, flag_sample=2, flag_augment=True):
#
#         self.root_dir = root_dir
#         self.transform_fnc = transform_fnc
#         if isinstance(img_size, tuple):
#             self.img_shape = img_size
#         else:
#             self.img_shape = (img_size, img_size)
#         self.flag_sample = flag_sample
#
#         self.root_img = 'clr/'
#         self.root_lndm = 'lndm/'
#         self.root_msk = 'msk/'
#         self.im_label, self.im_paths, self.im_index = [], [], []
#
#         self.flag_augment = flag_augment
#
#         if flag_init:
#             it_j=0
#             for it_i in range(label_num):
#
#
#                 imglist_all = [f for f in listdir(root_dir+self.root_lndm + str(it_i)) if isfile(join(root_dir, self.root_lndm + str(it_i), f)) and f[-4:] == ".jpg"]
#                 imglist_all_int = [int(x[:-4]) for x in imglist_all]
#                 imglist_all_int.sort()
#                 imglist_all = [(str(x).zfill(6) + ".jpg") for x in imglist_all_int]
#
#                 self.im_label += [it_i] * len(imglist_all)
#                 self.im_paths += imglist_all
#                 self.im_index += [it_j] * len(imglist_all)
#                 it_j+=1
#             print("Dataset initialized")
#             print("Data directory:", root_dir)  # 确认路径正确
#             print("Labels in dataset:", len(self.im_label))  # 应与样本数一致
#
#     def __len__(self):
#         return len(self.im_label)
#
#     def load_img(self, im_path):
#         im = Image.open(im_path)
#         im = im.resize([int(self.img_shape[0]*1.125)]*2, resample=Image.LANCZOS)
#         w, h = im.size
#
#         if self.flag_augment:
#             offset_h = 0.
#             center_h = h / 2 + offset_h * h
#             center_w = w / 2
#             min_sz, max_sz = w / 2, (w - center_w) * 1.5
#             diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2
#
#             img_res = im.crop(
#                 (int(center_w - crop_sz - diff_sz * self.crop_rnd[0]), int(center_h - crop_sz - diff_sz * self.crop_rnd[1]),
#                  int(center_w + crop_sz + diff_sz * self.crop_rnd[2]), int(center_h + crop_sz + diff_sz * self.crop_rnd[3])))
#         else:
#             offset_h = 0.
#             center_h = h / 2 + offset_h * h
#             center_w = w / 2
#             min_sz, max_sz = w / 2, (w - center_w) * 1.5
#             crop_sz = self.img_shape[0]/2
#             img_res = im.crop(
#                 (int(center_w - crop_sz),
#                  int(center_h - crop_sz),
#                  int(center_w + crop_sz),
#                  int(center_h + crop_sz)))
#
#         img_res = img_res.resize(self.img_shape, resample=Image.LANCZOS)
#         return self.transform_fnc(img_res)
#
#     def __getitem__(self, idx):
#         im_clr, im_lndm, im_msk, im_ind = [], [], [], []
#         if self.flag_sample==1:
#             idx = [idx]
#
#         for k_iter in range(self.flag_sample):
#             self.crop_rnd = [random.random(), random.random(), random.random(), random.random()]
#             im_clr_path = os.path.join(self.root_dir, self.root_img, str(self.im_label[idx[k_iter]]), self.im_paths[idx[k_iter]])
#             clr_img = self.load_img(im_clr_path)
#             im_clr.append(clr_img)
#
#             im_lndm_path = os.path.join(self.root_dir, self.root_lndm, str(self.im_label[idx[k_iter]]), self.im_paths[idx[k_iter]])
#             lndm_img = self.load_img(im_lndm_path)
#             im_lndm.append(lndm_img)
#
#             im_msk_path = os.path.join(self.root_dir, self.root_msk, str(self.im_label[idx[k_iter]]), self.im_paths[idx[k_iter]])
#             msk = ((1 - self.load_img(im_msk_path)) > 0.2)
#             im_msk.append(msk)
#
#             im_ind.append(self.im_index[idx[k_iter]])
#
#         return im_clr, im_lndm, im_msk, im_ind

from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 尝试容忍部分截断的图片

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_num=1200, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                 img_size = 128, flag_init=True, flag_sample=2, flag_augment=True):

        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        if isinstance(img_size, tuple):
            self.img_shape = img_size
        else:
            self.img_shape = (img_size, img_size)
        self.flag_sample = flag_sample

        self.root_img = 'clr/'
        self.root_lndm = 'lndm/'
        self.root_msk = 'msk/'
        self.im_label, self.im_paths, self.im_index = [], [], []

        self.flag_augment = flag_augment

        if flag_init:
            it_j = 0
            for it_i in range(label_num):
                imglist_all = [f for f in listdir(root_dir + self.root_lndm + str(it_i))
                               if isfile(join(root_dir, self.root_lndm + str(it_i), f)) and f[-4:] == ".jpg"]
                imglist_all_int = [int(x[:-4]) for x in imglist_all]
                imglist_all_int.sort(key=int)
                imglist_all = [(str(x).zfill(6) + ".jpg") for x in imglist_all_int]

                self.im_label += [it_i] * len(imglist_all)
                self.im_paths += imglist_all
                self.im_index += [it_j] * len(imglist_all)
                it_j += 1

            print("Dataset initialized")
            print("Data directory:", root_dir)
            print("Samples in dataset:", len(self.im_label))

    def __len__(self):
        return len(self.im_label)

    # def load_img(self, im_path):
    #     """只负责读 + 裁剪 + resize，不在这里处理异常"""
    #     im = Image.open(im_path)
    #     im = im.resize([int(self.img_shape[0] * 1.125)] * 2, resample=Image.LANCZOS)
    #     w, h = im.size
    #
    #     if self.flag_augment:
    #         offset_h = 0.
    #         center_h = h / 2 + offset_h * h
    #         center_w = w / 2
    #         min_sz, max_sz = w / 2, (w - center_w) * 1.5
    #         diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2
    #
    #         img_res = im.crop(
    #             (int(center_w - crop_sz - diff_sz * self.crop_rnd[0]),
    #              int(center_h - crop_sz - diff_sz * self.crop_rnd[1]),
    #              int(center_w + crop_sz + diff_sz * self.crop_rnd[2]),
    #              int(center_h + crop_sz + diff_sz * self.crop_rnd[3])))
    #     else:
    #         offset_h = 0.
    #         center_h = h / 2 + offset_h * h
    #         center_w = w / 2
    #         min_sz, max_sz = w / 2, (w - center_w) * 1.5
    #         crop_sz = self.img_shape[0] / 2
    #         img_res = im.crop(
    #             (int(center_w - crop_sz),
    #              int(center_h - crop_sz),
    #              int(center_w + crop_sz),
    #              int(center_h + crop_sz)))
    #
    #     img_res = img_res.resize(self.img_shape, resample=Image.LANCZOS)
    #     return self.transform_fnc(img_res)
    def load_img(self, im_path):
        """
        现在假设磁盘上的图已经是对齐好的 128x128（来自 get_lndm），
        这里只做：
        - 读图
        - 必要时做一次安全的 resize 到 self.img_shape
        - 转 tensor
        不再做随机裁剪等几何变换。
        """
        im = Image.open(im_path).convert("RGB")
        if im.size != self.img_shape:
            # 万一有少数图不是 128x128，就做一次安全 resize
            im = im.resize(self.img_shape, resample=Image.LANCZOS)
        return self.transform_fnc(im)

    def _get_single_sample(self, base_idx):
        """
        从某个 index 读取一条样本；如果遇到坏图/缺图，就随机换一个 index 继续尝试。
        """
        max_retry = 10
        idx = base_idx

        for _ in range(max_retry):
            label = self.im_label[idx]
            fname = self.im_paths[idx]

            self.crop_rnd = [random.random(), random.random(), random.random(), random.random()]

            clr_path = os.path.join(self.root_dir, self.root_img,  str(label), fname)
            lndm_path = os.path.join(self.root_dir, self.root_lndm, str(label), fname)
            msk_path = os.path.join(self.root_dir, self.root_msk,  str(label), fname)

            try:
                clr_img = self.load_img(clr_path)
                lndm_img = self.load_img(lndm_path)
                msk     = ((1 - self.load_img(msk_path)) > 0.2)
                ind     = self.im_index[idx]
                return clr_img, lndm_img, msk, ind
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                print(f"[WARN] skip bad sample {label}/{fname}: {e}")
                # 随机换一个 index 重新试
                idx = random.randint(0, len(self.im_label) - 1)

        # 连续很多次都失败，就抛出一个更清楚的错误
        raise RuntimeError(f"Too many invalid images around index {base_idx}")

    def __getitem__(self, idx):
        im_clr, im_lndm, im_msk, im_ind = [], [], [], []

        # 按你原来的逻辑，flag_sample == 1 时用单个 idx，其他情况假设外面传进来的就是一个 index 列表
        if self.flag_sample == 1:
            idx_list = [idx]
        else:
            idx_list = idx  # 保持原来的行为

        for k_iter in range(self.flag_sample):
            clr_img, lndm_img, msk, ind = self._get_single_sample(idx_list[k_iter])
            im_clr.append(clr_img)
            im_lndm.append(lndm_img)
            im_msk.append(msk)
            im_ind.append(ind)

        return im_clr, im_lndm, im_msk, im_ind


def load_data(DATA_PATH, DATA_SET, WORKERS_NUM, BATCH_SIZE, IMG_SIZE, FLAG_DATA_AUGM, LABEL_NUM, mode_train=True):
    ##### Data loaders
    data_dir = DATA_PATH + DATA_SET + '/'
    if mode_train:
        dataset_train = ImageDataset(root_dir=data_dir, label_num=LABEL_NUM, transform_fnc=transforms.Compose([transforms.ToTensor()]),
                                             img_size=IMG_SIZE, flag_augment = FLAG_DATA_AUGM)
        total_steps = int(len(dataset_train) / BATCH_SIZE)

        ddict = defaultdict(list)
        for idx, label in enumerate(dataset_train.im_label):
            ddict[label].append(idx)

        list_of_indices_for_each_class = []
        for key in ddict:
            list_of_indices_for_each_class.append(ddict[key])
        loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=False, sampler=SiameseSampler(list_of_indices_for_each_class, BATCH_SIZE, total_steps))
        # loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM,
        #                                            batch_size=BATCH_SIZE, shuffle=True)
        print("Total number of steps per epoch:", total_steps)
        print("Total number of training samples:", len(dataset_train))
        print("传入的 label_num:", LABEL_NUM)  # 确保是 1200
        # print("len(dataset_train):", len(dataset_train))
        # print("BATCH_SIZE:", BATCH_SIZE)
        # print("Computed total_steps:", total_steps)

        return loader_train, total_steps, LABEL_NUM
    else:
        label_num = 363
        dataset_test = ImageDataset(root_dir=data_dir, label_num=label_num,transform_fnc=transforms.Compose([transforms.ToTensor()]), img_size = IMG_SIZE)
        loader_test = torch.utils.data.DataLoader(dataset=dataset_test, num_workers=1, batch_size=1, shuffle=False)
        print("Total number of test samples:", len(dataset_test))
        return loader_test, len(dataset_test), label_num




# class SiameseSampler(Sampler):
#
#     def __init__(self, l_inds, batch_size, iterations_per_epoch):
#         self.l_inds = l_inds
#         self.max = -1
#         self.batch_size = batch_size
#         self.flat_list = []
#         self.iterations_per_epoch = iterations_per_epoch
#
#     def __iter__(self):
#         self.flat_list = []
#
#         for ii in range(int(self.iterations_per_epoch)):
#             # get half of the images randomly
#             sep = int(self.batch_size / 2)
#             for i in range(sep):
#                 first_class = random.choice(self.l_inds)
#                 second_class = random.choice(self.l_inds)
#                 first_element = random.choice(first_class)
#                 second_element = random.choice(second_class)
#                 self.flat_list.append([first_element, second_element])
#
#             # get the last half as images from util_data.pythe same class
#             for i in range(sep, self.batch_size):
#                 c_class = random.choice(self.l_inds)
#                 first_element = random.choice(c_class)
#                 second_element = random.choice(c_class)
#                 self.flat_list.append([first_element, second_element])
#
#         random.shuffle(self.flat_list)
#         return iter(self.flat_list)
#
#     def __len__(self):
#         return self.iterations_per_epoch * self.batch_size

class SiameseSampler(Sampler):

    def __init__(self, l_inds, batch_size, iterations_per_epoch):
        self.l_inds = l_inds               # 每个元素是同一类的 index 列表
        self.batch_size = batch_size
        self.flat_list = []
        self.iterations_per_epoch = iterations_per_epoch

    def __iter__(self):
        self.flat_list = []

        num_classes = len(self.l_inds)

        for _ in range(int(self.iterations_per_epoch)):
            for _ in range(self.batch_size):
                # 1) 随机选一个类作为“原图身份”
                c1_idx = random.randrange(num_classes)
                class1_indices = self.l_inds[c1_idx]

                # 2) 再选一个“不同”的类作为“目标身份”
                #    先在 [0, num_classes-2] 里选一个，再跳过 c1_idx
                c2_idx = random.randrange(num_classes - 1)
                if c2_idx >= c1_idx:
                    c2_idx += 1
                class2_indices = self.l_inds[c2_idx]

                # 3) 在两个类里各取一张图
                first_element = random.choice(class1_indices)   # image[0] 的 dataset index
                second_element = random.choice(class2_indices)  # image[1] 的 dataset index

                # 4) 存成一个 pair
                self.flat_list.append([first_element, second_element])

        # 打乱所有 pair 的顺序
        random.shuffle(self.flat_list)
        return iter(self.flat_list)

    def __len__(self):
        return self.iterations_per_epoch * self.batch_size

