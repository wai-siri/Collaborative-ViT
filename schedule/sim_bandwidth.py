import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from declining_rate import declining_rate
from schedule import schedule, init


def simulate_bandwidth(N, x_0, D_M, bits, num_steps, step, SLA):
    B_list_mbps = np.arange(50, 501, 25)  # 50, 75, 100, ..., 500 Mbps
    B_list_bps = B_list_mbps * 1000000      # 转为 bps

    alphas = []
    splits = []

    for B in B_list_bps:
        result = schedule(N, x_0, D_M, bits, num_steps, step, B, SLA)
        a, s = result
        alphas.append(a)
        splits.append(s)
        print(f"B={B/1e6} Mbps -> α={a}, s={s}")

    # 画图
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(BASE_DIR, '..', 'test_file', 'match_curve')

    # 图1: B -> α
    plt.plot(B_list_mbps, alphas, marker='o')
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('α (declining rate)')
    plt.title('Bandwidth vs α')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'bandwidth_vs_alpha.png'), dpi=150)
    plt.close()

    # 图2: B -> s
    plt.plot(B_list_mbps, splits, marker='o')
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('s (split point)')
    plt.title('Bandwidth vs Split Point')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'bandwidth_vs_split.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    model = init()
    N = len(model.blocks)  # model layer number
    x_0 = model.pos_embed.size(1)  # initial token number
    D_M = model.pos_embed.size(2)  # model token dim
    dtype = next(model.parameters()).dtype # token data type
    bits = torch.finfo(dtype).bits # 占用 bit 数

    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)
    SLA = 60.0

    simulate_bandwidth(N, x_0, D_M, bits, num_steps, step, SLA)