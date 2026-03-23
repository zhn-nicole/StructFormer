#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import dlib
import numpy as np

# ================== 需要修改的路径 ==================
SRC_DIR = r"/home/user/桌面/celeba/orig2"

# 处理后输出目录（会变成“和生成结果几何风格接近”的版本）
DST_DIR = r"/home/user/桌面/celeba/orig1"

# dlib 68 点模型所在目录（目录里要有 shape_predictor_68_face_landmarks.dat）
DLIB_DIR = r"/home/user/project/ciagan-master/source"

IMG_SIZE = 128

# 是否使用随机裁剪（对应老的 flag_augment）
USE_AUGMENT = False

# 缩小比例：0.9 稍微缩一点，0.8 更明显
SCALE_SHRINK = 1.5
# ==================================================


# ---------- STEP 1: 复刻 get_lndm 裁剪 ----------
def lndm_crop_to_128(img_bgr, detector, predictor, res_w=128, res_h=128):
    img = img_bgr.copy()
    img_dlib = np.asarray(img, dtype=np.uint8)

    dets = detector(img_dlib, 1)
    if len(dets) == 0:
        return None

    d = dets[0]
    landmarks = predictor(img_dlib, d)

    x39 = landmarks.part(39).x
    y39 = landmarks.part(39).y
    x42 = landmarks.part(42).x
    y42 = landmarks.part(42).y

    c_x = int((x42 + x39) / 2)
    c_y = int((y42 + y39) / 2)

    eye_dist = (x42 - x39)
    w_r = int(eye_dist * 4)
    h_r = int(eye_dist * 5)
    w_r = int(h_r / res_h * res_w)

    w = int(w_r * 2)
    h = int(h_r * 2)

    pd = int(w)
    img_p = cv2.copyMakeBorder(
        img, pd, pd, pd, pd, borderType=cv2.BORDER_REPLICATE
    )

    y1 = c_y - h_r + pd
    y2 = c_y + h_r + pd
    x1 = c_x - w_r + pd
    x2 = c_x + w_r + pd

    visual = img_p[y1:y2, x1:x2]
    visual = cv2.resize(visual, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
    return visual


# ---------- STEP 2: 复刻旧版 load_img 的放大 + 裁剪 ----------
def train_style_preprocess_cv2(img_bgr, img_size=128, augment=False, crop_rnd=None):
    big_sz = int(img_size * 1.125)  # 128 -> 144
    img_big = cv2.resize(img_bgr, (big_sz, big_sz), interpolation=cv2.INTER_LANCZOS4)

    h, w = img_big.shape[:2]
    center_h = h / 2
    center_w = w / 2
    min_sz, max_sz = w / 2, (w - center_w) * 1.5

    if augment:
        if crop_rnd is None:
            crop_rnd = np.random.rand(4)
        else:
            crop_rnd = np.asarray(crop_rnd, dtype=float)

        diff_sz = (max_sz - min_sz) / 2.0
        crop_sz = min_sz / 2.0

        left   = int(center_w - crop_sz - diff_sz * crop_rnd[0])
        top    = int(center_h - crop_sz - diff_sz * crop_rnd[1])
        right  = int(center_w + crop_sz + diff_sz * crop_rnd[2])
        bottom = int(center_h + crop_sz + diff_sz * crop_rnd[3])
    else:
        crop_sz = img_size / 2.0
        left   = int(center_w - crop_sz)
        top    = int(center_h - crop_sz)
        right  = int(center_w + crop_sz)
        bottom = int(center_h + crop_sz)

    img_crop = img_big[top:bottom, left:right]
    img_out = cv2.resize(img_crop, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    return img_out


# ---------- STEP 3: 缩小人脸（整体缩小 + 居中填充） ----------
def shrink_whole_image_center(img_bgr, img_size=128, scale=0.85):
    """
    把整张 128x128 图缩小到 (scale * 128)，再居中贴回 128x128 画布。
    scale < 1 -> 脸变小，背景边缘多一些。
    """
    h, w = img_bgr.shape[:2]
    assert h == img_size and w == img_size

    new_size = max(1, int(img_size * scale))
    img_small = cv2.resize(img_bgr, (new_size, new_size), interpolation=cv2.INTER_LANCZOS4)

    # 先用边界像素扩展出一张 128x128 画布
    canvas = cv2.copyMakeBorder(
        img_small,
        top=max(0, (img_size - new_size) // 2),
        bottom=max(0, img_size - new_size - (img_size - new_size) // 2),
        left=max(0, (img_size - new_size) // 2),
        right=max(0, img_size - new_size - (img_size - new_size) // 2),
        borderType=cv2.BORDER_REPLICATE
    )

    # 再确保是精确 128x128（防止整数除法带来 1 像素误差）
    canvas = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)
    return canvas


# ---------- 整体 pipeline ----------
def process_one_image(path_in, path_out, detector, predictor):
    img = cv2.imread(path_in)
    if img is None:
        print(f"[WARN] 无法读取: {path_in}")
        return False

    # step 1: get_lndm 风格裁剪
    clr_like = lndm_crop_to_128(img, detector, predictor, res_w=IMG_SIZE, res_h=IMG_SIZE)
    if clr_like is None:
        print(f"[WARN] 未检测到人脸: {path_in}")
        return False

    # step 2: 训练时放大裁剪
    mid = train_style_preprocess_cv2(clr_like, img_size=IMG_SIZE, augment=USE_AUGMENT)

    # step 3: 整体缩小一点（让脸看起来没那么大）
    final_img = shrink_whole_image_center(mid, img_size=IMG_SIZE, scale=SCALE_SHRINK)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    cv2.imwrite(path_out, final_img)
    return True


def process_dir():
    predictor_path = os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")
    if not os.path.isfile(predictor_path):
        raise FileNotFoundError(f"未找到 dlib 模型: {predictor_path}")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(os.listdir(SRC_DIR))
    total = 0
    ok = 0

    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        total += 1
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)

        if process_one_image(src, dst, detector, predictor):
            ok += 1

        if total % 20 == 0:
            print(f"[INFO] 已尝试处理 {total} 张，成功 {ok} 张")

    print(f"[DONE] 总共尝试 {total} 张，成功 {ok} 张，输出目录: {DST_DIR}")


if __name__ == "__main__":
    process_dir()
