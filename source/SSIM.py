import os
from os.path import join, splitext
import numpy as np
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim


def compute_ssim_folder(
    orig_dir,
    gen_dir,
    exts=(".jpg", ".jpeg", ".png", ".bmp"),
):
    """
    对两个文件夹中一一对应的图像计算 SSIM
    orig_dir: 原始图像目录
    gen_dir : 生成图像目录
    exts    : 允许的扩展名
    """
    exts = tuple(e.lower() for e in exts)

    # 找到原始目录里的文件列表
    all_files = [
        f for f in os.listdir(orig_dir)
        if splitext(f)[1].lower() in exts
    ]
    all_files.sort()

    scores = []

    missing = 0
    for fname in tqdm(all_files, desc="Computing SSIM"):
        orig_path = join(orig_dir, fname)
        gen_path  = join(gen_dir, fname)

        if not os.path.exists(gen_path):
            print(f"[WARN] generated file not found, skip: {fname}")
            missing += 1
            continue

        # 读图并转为 RGB
        orig_img = Image.open(orig_path).convert("RGB")
        gen_img  = Image.open(gen_path).convert("RGB")

        # 如果大小不一致，把生成图 resize 到原图大小
        if orig_img.size != gen_img.size:
            gen_img = gen_img.resize(orig_img.size, Image.BILINEAR)

        orig_np = np.array(orig_img)
        gen_np  = np.array(gen_img)

        # skimage 的 ssim：新版用 channel_axis=-1，旧版用 multichannel=True
        try:
            score = ssim(
                orig_np,
                gen_np,
                channel_axis=-1,   # 对应多通道 RGB
                data_range=255
            )
        except TypeError:
            # 如果你的 scikit-image 比较旧，没有 channel_axis 参数
            score = ssim(
                orig_np,
                gen_np,
                multichannel=True,
                data_range=255
            )

        scores.append(score)

    scores = np.array(scores)
    print("======================================")
    print(f"有效样本数: {len(scores)}  (缺失: {missing})")
    if len(scores) > 0:
        print(f"SSIM 均值 : {scores.mean():.4f}")
        print(f"SSIM 方差 : {scores.var():.4f}")
        print(f"SSIM 最小值: {scores.min():.4f}")
        print(f"SSIM 最大值: {scores.max():.4f}")
    print("======================================")


if __name__ == "__main__":
    # TODO: 把这里的路径改成你当前要评估的那一对文件夹
    orig_dir = "/home/user/project/ciagan-master/celeba128/merged"   # 原图
    gen_dir  = "/home/user/project/ciagan-master/12-6"              # 生成图

    compute_ssim_folder(orig_dir, gen_dir)
