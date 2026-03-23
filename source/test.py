import os
import time  # <<< 新增：计时
import torch
import numpy as np
import util_func
import util_data
from torchvision import transforms, utils
from os.path import join
from os import listdir
import arch.arch_unet_flex as arch_gen
import argparse

def _load_state_dict_strict(model, sd):
    # 兼容 DataParallel 保存的 'module.' 前缀
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        nk = k.replace('module.', '') if k.startswith('module.') else k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=True)

#在测试数据上进行推理，生成匿名化的人脸图像并保存
def inference(generator, out_dir, data_loader, device_comp, num_classes = 1200):
    total_imgs = 0

    # --------- 计时相关变量（用于论文统计）---------
    iter_times = []      # 每张图端到端时间（读数据 + 前向 + 后处理 + 存盘）
    net_times = []       # 每张图仅模型前向时间
    t_dataset_start = time.perf_counter()

    use_cuda = (device_comp.type == 'cuda')

    for batch in data_loader:  # 遍历测试数据加载器，逐批处理数据。batch_size=1

        # ====== 单张图端到端开始计时 ======
        if use_cuda:
            torch.cuda.synchronize()
        t_iter_start = time.perf_counter()

        # prepare data
        # batch 包含 im_faces（人脸）、im_lndm（关键点）、im_msk（蒙板）、im_ind（身份索引）。
        im_faces, im_lndm, im_msk, im_ind = [item[0].float().to(device_comp) for item in batch]

        output_id = (int(im_ind[0].cpu())+1) % num_classes  # chose next id 选择目标身份的索引。

        labels_one_hot = np.zeros((1, num_classes))  # 为目标身份创建 one-hot 编码标签。
        labels_one_hot[0, output_id] = 1
        labels_one_hot = torch.tensor(labels_one_hot).float().to(device_comp)

        # ====== 模型前向开始计时（仅网络本身） ======
        if use_cuda:
            torch.cuda.synchronize()
        t_net_start = time.perf_counter()

        # inference
        with torch.no_grad():  # 禁用梯度计算。
            input_gen = im_faces[0] * (1 - im_msk[0])
            im_gen = generator(input_gen, im_lndm, onehot=labels_one_hot)

            # 保存生成器直接输出的图像（如果需要单独分析，可重新打开）
            # gen_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
            # gen_out.save(join(out_dir, f"gen_{str(total_imgs).zfill(6)}.jpg"))

            im_gen = torch.clamp(im_gen * im_msk + im_faces * (1 - im_msk), 0, 1)  # final image with BG

        if use_cuda:
            torch.cuda.synchronize()
        t_net_end = time.perf_counter()
        net_times.append(t_net_end - t_net_start)

        # ====== 后处理 + 存盘（仍算在端到端时间里） ======
        img_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
        img_out.save(join(out_dir, str(total_imgs).zfill(6) + '.jpg'))
        total_imgs += 1

        # ====== 单张图端到端结束计时 ======
        if use_cuda:
            torch.cuda.synchronize()
        t_iter_end = time.perf_counter()
        iter_times.append(t_iter_end - t_iter_start)

    # ====== 整个数据集计时结束 ======
    if use_cuda:
        torch.cuda.synchronize()
    t_dataset_end = time.perf_counter()
    total_time = t_dataset_end - t_dataset_start

    print("Done.")
    # --------- 统计并打印可写进论文的指标 ---------
    iter_times = np.array(iter_times, dtype=np.float64)
    net_times = np.array(net_times, dtype=np.float64)

    # 避免空数据集出错
    if total_imgs == 0:
        print("Warning: no images were processed.")
        return

    # 平均延迟（ms / image）
    avg_iter_ms = iter_times.mean() * 1000.0
    avg_net_ms = net_times.mean() * 1000.0

    # 吞吐量（images / s）
    throughput = total_imgs / total_time if total_time > 0 else float('nan')

    # 分位数（ms）
    def pct(arr, p):
        return np.percentile(arr, p) * 1000.0

    print("\n====== Inference Speed Statistics ======")
    print(f"Number of images          : {total_imgs}")
    print(f"Total wall-clock time     : {total_time:.4f} s")
    print(f"Average latency (end-to-end, incl. I/O): {avg_iter_ms:.3f} ms / image")
    print(f"Average latency (network forward only): {avg_net_ms:.3f} ms / image")
    print(f"Throughput (end-to-end)   : {throughput:.3f} images / s")

    print("\nPer-image latency percentiles (end-to-end):")
    print(f"  P50 (median)            : {pct(iter_times, 50):.3f} ms")
    print(f"  P90                     : {pct(iter_times, 90):.3f} ms")
    print(f"  P95                     : {pct(iter_times, 95):.3f} ms")
    print(f"  P99                     : {pct(iter_times, 99):.3f} ms")

    print("\nPer-image latency percentiles (network forward only):")
    print(f"  P50 (median)            : {pct(net_times, 50):.3f} ms")
    print(f"  P90                     : {pct(net_times, 90):.3f} ms")
    print(f"  P95                     : {pct(net_times, 95):.3f} ms")
    print(f"  P99                     : {pct(net_times, 99):.3f} ms")
    print("========================================\n")


def run_inference(data_path='../dataset/celeba/', num_folders = -1, model_path = '../Aunet_flex_Dceleba_Tcheck_ep00050G', output_path = '../10-15',use_ema=True, ema_path=None):
    ##### PREPARING DATA
    if num_folders==-1:
        num_folders = len(listdir(join(data_path,'lndm')))

    dataset_test = util_data.ImageDataset(
        root_dir=data_path,
        label_num=num_folders,
        transform_fnc=transforms.Compose([transforms.ToTensor()]),
        flag_sample=1,
        flag_augment=False
    )
    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    ##### PREPARING MODELS
    device_comp = util_func.set_comp_device(True)
    model = arch_gen.Generator().to(device_comp)
    model.eval()

    # 常规权重路径（不带 .pth 后缀时我们手动补）
    g_path = model_path if model_path.endswith('.pth') else (model_path + '.pth')

    # EMA 路径（默认把 _G_ema.pth 放在同目录）
    if ema_path is None:
        # 假设 model_path 形如 ".../NAME_ep00050G.pth" 或 ".../NAME_G.pth"
        if g_path.endswith('_G.pth'):
            ema_guess = g_path.replace('_G.pth', '_G_ema.pth')
        else:
            ema_guess = g_path.replace('.pth', '_ema.pth')  # 兜底
        ema_path = ema_guess

    # 优先加载 EMA；失败再退回普通
    sd_loaded = False
    if use_ema and os.path.exists(ema_path):
        try:
            ema_sd = torch.load(ema_path, map_location='cpu')
            _load_state_dict_strict(model, ema_sd)
            print(f'[EMA] Loaded EMA weights: {ema_path}')
            sd_loaded = True
        except Exception as e:
            print(f'[EMA] Failed to load EMA weights ({ema_path}): {e}')

    if not sd_loaded:
        print(f'[EMA] EMA not used; fallback to: {g_path}')
        _load_state_dict_strict(model, torch.load(g_path, map_location='cpu'))

    print('Model is ready (eval mode).')
    inference(model, output_path, data_loader, device_comp=device_comp)

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to input data directory', default='../dataset/celeba/')
parser.add_argument('--ids', type=int, help='how many folder/ids to process', default=1200)
parser.add_argument('--model', type=str, help='path to a pre-trained model and its name', default='../models-2/ciagan_Aunet_flex_Dceleba_Tcheck/Aunet_flex_Dceleba_Tcheck_ep00050G')
parser.add_argument('--out', type=str, help='path to output data directory', default='../12-17')
parser.add_argument('--use-ema', action='store_true', help='use EMA weights if available')
parser.add_argument('--ema-path', type=str,  help='explicit path to EMA weights (optional)', default='../models-4/ciagan_Aunet_flex_Dceleba_Tcheck/Aunet_flex_Dceleba_Tcheck_ep00050G_ema.pth')

args = parser.parse_args()

run_inference(
    data_path=args.data,
    num_folders=args.ids,
    model_path=args.model,
    output_path=args.out,
    use_ema=args.use_ema,
    ema_path=args.ema_path
)
