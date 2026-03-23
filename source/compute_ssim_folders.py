import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ====== 配置区域：改成你自己的路径 ======
DIR_REF = "/home/user/project/ciagan-master/celeba128/merged"  # 原始图像
DIR_GEN = "/home/user/project/ciagan-master/11-26"             # 生成图像
OUT_FILE = "ssim_celeba_11-26.txt"
# ===================================

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def main():
    ref_files = sorted([f for f in os.listdir(DIR_REF) if f.lower().endswith(EXTS)])
    gen_files = sorted([f for f in os.listdir(DIR_GEN) if f.lower().endswith(EXTS)])

    n = min(len(ref_files), len(gen_files))
    ref_files = ref_files[:n]
    gen_files = gen_files[:n]

    print(f"pairs: {n}")

    ssim_values = []

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("# idx\tref\tgen\tssim\n")

        for i in range(n):
            name_ref = ref_files[i]
            name_gen = gen_files[i]

            p_ref = os.path.join(DIR_REF, name_ref)
            p_gen = os.path.join(DIR_GEN, name_gen)

            img_ref = load_image(p_ref)
            img_gen = load_image(p_gen)

            # 确保大小一致
            if img_ref.size != img_gen.size:
                img_gen = img_gen.resize(img_ref.size, Image.BILINEAR)

            arr_ref = np.array(img_ref, dtype=np.float32) / 255.0
            arr_gen = np.array(img_gen, dtype=np.float32) / 255.0

            # skimage 旧版本用 multichannel，新版本会有警告但还能用
            val = ssim(arr_ref, arr_gen, data_range=1.0, multichannel=True)
            ssim_values.append(val)

            if (i + 1) % 500 == 0 or i == n - 1:
                print(f"{i+1}/{n}  current SSIM = {val:.4f}")

            f.write(f"{i}\t{name_ref}\t{name_gen}\t{val:.6f}\n")

    ssim_values = np.array(ssim_values, dtype=np.float32)

    mean_ssim = float(ssim_values.mean())
    std_ssim = float(ssim_values.std(ddof=1))
    min_ssim = float(ssim_values.min())
    max_ssim = float(ssim_values.max())

    print("==== SSIM stats ====")
    print("N =", n)
    print("mean =", mean_ssim)
    print("std  =", std_ssim)
    print("min  =", min_ssim)
    print("max  =", max_ssim)

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n# N={n}\n")
        f.write(f"# mean={mean_ssim:.6f}\n")
        f.write(f"# std={std_ssim:.6f}\n")
        f.write(f"# min={min_ssim:.6f}\n")
        f.write(f"# max={max_ssim:.6f}\n")


if __name__ == "__main__":
    main()
