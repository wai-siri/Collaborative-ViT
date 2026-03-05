import json
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_large_patch16_384', pretrained=False)
model.eval()
model = model.to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def paint(step, start_tokens, k, b, num_tokens, latencies, _):
    RESULT_DIR = os.path.join(BASE_DIR, '..', 'test_file', 'result')

    x_line = np.linspace(step, start_tokens, 100)
    y_line = k * x_line + b

    plt.scatter(num_tokens, latencies)
    plt.plot(x_line, y_line)
    plt.xlabel('Number of Tokens')
    plt.ylabel('latencies (ms)')
    plt.savefig(os.path.join(RESULT_DIR, f'latency_curve_{_}.png'), dpi=150)
    plt.close()


def warm_up(img):
    for _ in range(10):
        model.blocks[0](img)

def measurement(img, block_idx):
    epoch = 5
    torch.cuda.synchronize() # 确保之前的所有操作都已完成
    start_time = time.perf_counter()
    for _ in range(epoch):
        model.blocks[block_idx](img)
    torch.cuda.synchronize() # 再次同步，确保操作完成
    end_time = time.perf_counter()
    return (end_time - start_time) / epoch

def test_time_main(start_tokens, step, dim, block_idx):
    results = {}

    with torch.no_grad(): # 关闭梯度计算，减少额外开销
        for num in range(start_tokens, 0, -step):
            tokens = torch.randn(1, num, dim).to(device)
            latency = measurement(tokens, block_idx)
            results[num] = latency

    return results

if __name__ == "__main__":
    N = len(model.blocks)
    start_tokens = model.pos_embed.size(1)
    dim = model.pos_embed.size(2)  # 隐藏层维度

    tokens = torch.randn(1, start_tokens, dim).to(device)
    with torch.no_grad():
        warm_up(tokens)

    epoch, step = 5, 1 # 单层执行轮数
    all_layers = {}
    for block_idx in range(N):
        sum_k, sum_b = 0, 0
        for _ in range(epoch):
            results = test_time_main(start_tokens, step, dim, block_idx)

            num_tokens = np.array(list(results.keys()))
            latencies = np.array(list(results.values()))
            latencies *= 1000

            k, b = np.polyfit(num_tokens, latencies, deg=1)
            print(f"block {block_idx}, round {_}: k={k}, b={b}")
            if block_idx == 0: paint(step, start_tokens, k, b, num_tokens, latencies, _)
            sum_k += k
            sum_b += b
        ave_k = sum_k / epoch
        ave_b = sum_b / epoch
        all_layers[str(block_idx)] = {"k": ave_k, "b": ave_b}
        print(f"block {block_idx}: ave_k={ave_k}, ave_b={ave_b}")

    # 保存到 JSON 文件
    all_layers["model"] = "vit_large_patch16_384"
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'assets', 'profiler_k_b')
    with open(os.path.join(OUTPUT_DIR, 'device_k_b.json'), 'w') as f:
        json.dump(all_layers, f, indent=4)
