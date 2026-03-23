import json
import math
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm
import torch

from schedule.declining_rate import declining_rate
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 从 src/schedule/ 向上两级到达项目根目录
ASSETS_DIR = os.path.join(BASE_DIR, '..', '..', 'assets', 'profiler_k_b')

with open(os.path.join(ASSETS_DIR, 'device_k_b.json'), 'r') as f:
    device_profiler_data = json.load(f)
with open(os.path.join(ASSETS_DIR, 'cloud_k_b.json'), 'r') as f:
    cloud_profiler_data = json.load(f)

def init():
    if config.LOCAL_CHECKPOINT_PATH:
        # 离线模式：先创建空模型，再加载本地权重
        print(f"[Init] Loading model from local checkpoint: {config.LOCAL_CHECKPOINT_PATH}")
        model = timm.create_model(config.MODEL_NAME, pretrained=False)
        model.load_pretrained(config.LOCAL_CHECKPOINT_PATH)
    else:
        # 在线模式：由 timm 自动下载
        print(f"[Init] Downloading model from online...")
        model = timm.create_model(config.MODEL_NAME, pretrained=True)
    model.eval()
    model = model.to(device)
    return model

def device_profiler(x_l, layer):
    k = device_profiler_data[str(layer - 1)]["k"]  # layer 从1开始，block_idx 从0开始
    b = device_profiler_data[str(layer - 1)]["b"]
    return k * x_l + b

def cloud_profiler(x_l, layer):
    k = cloud_profiler_data[str(layer - 1)]["k"]
    b = cloud_profiler_data[str(layer - 1)]["b"]
    return k * x_l + b

def schedule(N, x_0, D_M, bits, num_steps, step, B, SLA, split_k=5):
    # 全局最优 fallback：当没有任何配置满足 SLA 时，返回总时延最低的配置
    best_fail_total_ms = float('inf')
    best_fail_alpha = 0.0
    best_fail_split = 0

    for i in range(num_steps):
        a = i * step
        # 更新出 {x_l | l = 1 ... N}: number of tokens at layer l
        x_l = {}
        tx0 = x_0
        if a == 0:
            for l in range(1, N + 1): x_l[l]=x_0
        else:
            for l in range(1, N + 1):
                delta_x_l = math.floor(2 ** (a * (N - (l - 1))))
                tx0 = tx0 - delta_x_l
                x_l[l]=max(tx0, 1)

        T_device, T_cloud, T_comm = {}, {}, {}
        T_comm[N + 1] = 0.0  # device-only: 无通信
        T_comm[0] = (x_0 * D_M * bits) / B * 1000  # cloud-only: 传输初始 token 表示

        # fine-to-coarse 候选 split point 集合
        C = set()
        C_s = 1
        C.add(C_s)
        for j in range(2, N + 1):
            C_s += math.ceil(j / split_k)
            if C_s > N: break
            C.add(C_s)

        for l in range(1, N + 1):
            T_device[l] = device_profiler(x_l[l], l)
            T_cloud[l] = cloud_profiler(x_l[l], l)
            if l in C:
                T_comm[l] = (x_l[l] * D_M * bits) / B * 1000

        # 加入边界候选点：0 (cloud-only) 和 N+1 (device-only)
        C.add(0)
        C.add(N + 1)

        # 滑动窗口计算 device_sum 和 cloud_sum
        t_device_sum, t_cloud_sum = 0.0, 0.0
        for j in range(1, N + 1): t_cloud_sum += T_cloud[j]

        L_sa, s_ans = SLA + 1, -1
        ls = 1
        for s in range(0, N + 2):
            while(ls <= s and s <= N):
                t_device_sum += T_device[ls]
                t_cloud_sum -= T_cloud[ls]
                ls += 1
            if s in C and t_device_sum + t_cloud_sum + T_comm[s] < L_sa:
                L_sa = t_device_sum + t_cloud_sum + T_comm[s]
                s_ans = s

        # 更新全局最优 fallback
        if s_ans >= 0 and L_sa < best_fail_total_ms:
            best_fail_total_ms = L_sa
            best_fail_alpha = a
            best_fail_split = s_ans

        # 如果当前 alpha 的最优 split 满足 SLA，立即返回
        if L_sa <= SLA:
            return a, s_ans

    # 没有任何配置满足 SLA → 返回全局总时延最低的配置
    return best_fail_alpha, best_fail_split

if __name__ == '__main__':
    model = init()
    N = len(model.blocks)  # model layer number
    x_0 = model.pos_embed.size(1)  # initial token number
    D_M = model.pos_embed.size(2)  # model token dim
    dtype = next(model.parameters()).dtype # token data type
    bits = torch.finfo(dtype).bits # 占用 bit 数

    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step) + 1
    B = 210 * 1000000  # bps(bits/s) 200-300
    SLA = 100.0

    print(schedule(N, x_0, D_M, bits, num_steps, step, B, SLA))