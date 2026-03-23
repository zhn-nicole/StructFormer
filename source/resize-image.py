import os
import cv2
import numpy as np
import argparse
from os import listdir, mkdir
from os.path import isfile, join, isdir, splitext


def resize_and_crop_celeba_images(input_path, output_path, target_size=(128, 128), jpeg_quality=90):
    """
    将 CelebA 数据集中的图片裁剪并调整为指定的目标分辨率，并将 PNG 格式转换为 JPG 格式。
    CelebA 原始图片分辨率为 178x218 像素。为了将其调整为 128x128，
    通常会先从中心裁剪一个正方形区域，然后缩放。
    这里采用从中心裁剪 178x178 区域，然后缩放到 128x128 的策略。

    参数:
    input_path (str): CelebA 原始图片所在的根目录路径。
                      该目录下应包含多个子文件夹，每个子文件夹代表一个身份，
                      其中包含该身份的多张图片。
    output_path (str): 处理后的图片将保存到的根目录路径。
                       脚本将在此路径下创建与输入身份文件夹对应的子文件夹。
    target_size (tuple): 目标图像的宽度和高度，例如 (128, 128)。
    jpeg_quality (int): 输出 JPEG 图像的质量，范围 0-100。值越高，质量越好，文件越大。
    """

    target_width, target_height = target_size

    # 确保输出根目录存在
    if not isdir(output_path):
        mkdir(output_path)
        print(f"创建输出目录: {output_path}")

    # 获取所有身份文件夹列表并排序
    # 确保只处理目录，跳过文件
    identity_folders = [f for f in listdir(input_path) if isdir(join(input_path, f))]
    identity_folders.sort()

    print(f"开始处理 {len(identity_folders)} 个身份文件夹...")

    # 遍历每个身份文件夹
    for fld in identity_folders:
        current_input_folder = join(input_path, fld)
        current_output_folder = join(output_path, fld)

        # 为当前身份创建输出子目录（如果不存在）
        if not isdir(current_output_folder):
            mkdir(current_output_folder)

        # 修正：根据您的描述，输入文件是 PNG 格式，因此将文件过滤条件从 ".jpg" 改为 ".png"。
        image_files = [f for f in listdir(current_input_folder) if
                       isfile(join(current_input_folder, f)) and f.lower().endswith(".jpg")]
        image_files.sort(key=lambda x: int(splitext(x)[0]))  # 假设文件名是数字，按数字排序

        print(f"  正在处理身份文件夹: {fld} ({len(image_files)} 张图片)")

        # 遍历当前身份文件夹中的每张图片
        for img_name in image_files:
            full_img_path = join(current_input_folder, img_name)

            # 修正：获取不带扩展名的文件名，并将其输出扩展名改为.jpg。
            base_name, _ = splitext(img_name)
            output_img_name = f"{base_name}.jpg"
            output_img_path = join(current_output_folder, output_img_name)

            # 读取图片
            img = cv2.imread(full_img_path)

            if img is None:
                print(f"    警告: 无法读取图像 {full_img_path}，跳过。")
                continue

            # 获取原始图像尺寸
            h, w, _ = img.shape  # CelebA 原始尺寸通常是 218x178

            # 计算裁剪区域：从中心裁剪一个正方形区域
            # 原始图片是 218 (高) x 178 (宽)
            # 目标裁剪尺寸是 178x178
            start_y = (h - w) // 2  # (218 - 178) // 2 = 40 // 2 = 20
            end_y = start_y + w  # 20 + 178 = 198

            # 执行中心裁剪
            cropped_img = img[start_y:end_y, 0:w]  # 裁剪为 178x178

            # 调整裁剪后图像的大小到目标分辨率 (128x128)
            # INTER_AREA 适用于缩小图像，INTER_CUBIC 适用于放大图像
            resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # 修正：保存处理后的图片为 JPG 格式，并应用指定的质量参数。
            cv2.imwrite(output_img_path, resized_img, )

        print(f"  身份文件夹 {fld} 处理完成。")

    print("所有图片处理完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将 CelebA 图片裁剪并调整为指定分辨率，并将 PNG 转换为 JPG。')
    parser.add_argument('--input_dir', type=str,
                        help='包含身份文件夹的输入数据目录路径（预期为 PNG 图片）',
                        default='../dataset/celeba/orig/')
    parser.add_argument('--output_dir', type=str,
                        help='处理后的图片将保存到的目录路径（输出为 JPG 图片）',
                        default='../dataset/celeba/orig1/')
    parser.add_argument('--target_width', type=int,
                        help='目标图像宽度', default=128)
    parser.add_argument('--target_height', type=int,
                        help='目标图像高度', default=128)
    # 新增 JPEG 质量参数，允许用户控制输出 JPG 的质量。
    parser.add_argument('--jpeg_quality', type=int, default=90,
                        help='输出 JPEG 图像的质量 (0-100)。值越高，质量越好，文件越大。')

    args = parser.parse_args()

    # 调用主处理函数，并传递新的质量参数
    resize_and_crop_celeba_images(
        input_path=args.input_dir,
        output_path=args.output_dir,
        target_size=(args.target_width, args.target_height),
        jpeg_quality=args.jpeg_quality
    )

