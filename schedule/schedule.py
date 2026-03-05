import json
import math
import os

import timm
import torch

from declining_rate import declining_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, '..', 'assets', 'profiler_k_b')
with open(os.path.join(ASSETS_DIR, 'device_k_b.json'), 'r') as f:
    device_profiler_data = json.load(f)
with open(os.path.join(ASSETS_DIR, 'cloud_k_b.json'), 'r') as f:
    cloud_profiler_data = json.load(f)

def init():
    model = timm.create_model('vit_large_patch16_384', pretrained=False)
    model.eval()
    model = model.to(device)
    return model

def device_profiler(x_l, layer): # test_time 输出
    # ave_k= 0.008612546224559241 ave_b= 0.3093493317437366
    k = device_profiler_data[str(layer - 1)]["k"]  # layer 从1开始，block_idx 从0开始
    b = device_profiler_data[str(layer - 1)]["b"]
    return k * x_l + b

def cloud_profiler(x_l, layer): # 虚拟数据
    k = cloud_profiler_data[str(layer - 1)]["k"]
    b = cloud_profiler_data[str(layer - 1)]["b"]
    return k * x_l + b

def schedule(N, x_0, D_M, bits, num_steps, step, B, SLA):
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
                x_l[l]=tx0

        T_device, T_cloud, T_comm = {}, {}, {}
        T_comm[N + 1] = 0.0
        T_comm[0] = (x_0 * D_M * bits) / B * 1000

        C, C_k = set(), 5 # k 设置为 3
        C_s = 1
        C.add(C_s)
        for j in range(2, N + 1):
            C_s += math.ceil(j / C_k)
            if C_s > N: break
            C.add(C_s)

        for l in range(1, N + 1):
            T_device[l] = device_profiler(x_l[l], l)
            T_cloud[l] = cloud_profiler(x_l[l], l)
            if l in C:
                T_comm[l] = (x_l[l] * D_M * bits) / B * 1000
                # print(T_comm[l], x_l[l], D_M, bits, B)

        C.add(0)
        C.add(N + 1)

        # 预更新
        t_device_sum, t_cloud_sum = 0.0, 0.0
        for j in range(1, N + 1): t_cloud_sum += T_cloud[j]

        L_sa, s_ans = SLA + 1, -1
        ls = 1
        for s in range(0, N + 2):
            while(ls <= s and s <= N):
                t_device_sum += T_device[ls]
                t_cloud_sum -= T_cloud[ls]
                ls += 1
            # if s in C: print("s in c:", t_device_sum, t_cloud_sum, T_comm[s])
            if s in C and t_device_sum + t_cloud_sum + T_comm[s] < L_sa:
                print("new_choice:", t_device_sum, t_cloud_sum, T_comm[s])
                L_sa = t_device_sum + t_cloud_sum + T_comm[s]
                s_ans = s
        if L_sa <= SLA:
            return a, s_ans
    return num_steps * step, 0

if __name__ == '__main__':
    model = init()
    N = len(model.blocks)  # model layer number
    x_0 = model.pos_embed.size(1)  # initial token number
    D_M = model.pos_embed.size(2)  # model token dim
    dtype = next(model.parameters()).dtype # token data type
    bits = torch.finfo(dtype).bits # 占用 bit 数

    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)
    B = 210 * 1000000  # bps(bits/s) 200-300
    SLA = 100.0

    print(schedule(N, x_0, D_M, bits, num_steps, step, B, SLA))