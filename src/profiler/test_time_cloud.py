"""
云端GPU Profiler数据采集脚本
用于在AutoDL云服务器上采集ViT各层的延迟数据
与test_time.py逻辑相同，但输出到cloud_k_b.json
"""

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
    """绘制延迟曲线"""
    # 从 src/profiler/ 向上两级到达项目根目录
    RESULT_DIR = os.path.join(BASE_DIR, '..', '..', 'test_file', 'result')
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    x_line = np.linspace(step, start_tokens, 100)
    y_line = k * x_line + b

    plt.scatter(num_tokens, latencies)
    plt.plot(x_line, y_line)
    plt.xlabel('Number of Tokens')
    plt.ylabel('latencies (ms)')
    plt.title(f'Cloud GPU Latency Curve (Round {_})')
    plt.savefig(os.path.join(RESULT_DIR, f'cloud_latency_curve_{_}.png'), dpi=150)
    plt.close()


def warm_up(img):
    """预热GPU"""
    for _ in range(10):
        model.blocks[0](img)


def measurement(img, block_idx):
    """测量单层延迟"""
    epoch = 5
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(epoch):
        model.blocks[block_idx](img)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / epoch


def test_time_main(start_tokens, step, dim, block_idx):
    """测试不同token数量下的延迟"""
    results = {}

    with torch.no_grad():
        for num in range(start_tokens, 0, -step):
            tokens = torch.randn(1, num, dim).to(device)
            latency = measurement(tokens, block_idx)
            results[num] = latency

    return results


if __name__ == "__main__":
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    N = len(model.blocks)
    start_tokens = model.pos_embed.size(1)
    dim = model.pos_embed.size(2)

    tokens = torch.randn(1, start_tokens, dim).to(device)
    with torch.no_grad():
        warm_up(tokens)

    epoch, step = 5, 1
    all_layers = {}
    
    print(f"\nProfiling {N} layers...")
    for block_idx in range(N):
        sum_k, sum_b = 0, 0
        for round_idx in range(epoch):
            results = test_time_main(start_tokens, step, dim, block_idx)

            num_tokens = np.array(list(results.keys()))
            latencies = np.array(list(results.values()))
            latencies *= 1000  # 转换为ms

            k, b = np.polyfit(num_tokens, latencies, deg=1)
            print(f"block {block_idx}, round {round_idx}: k={k:.6f}, b={b:.6f}")
            
            if block_idx == 0:
                paint(step, start_tokens, k, b, num_tokens, latencies, round_idx)
            
            sum_k += k
            sum_b += b
        
        ave_k = sum_k / epoch
        ave_b = sum_b / epoch
        all_layers[str(block_idx)] = {"k": ave_k, "b": ave_b}
        print(f"block {block_idx}: ave_k={ave_k:.6f}, ave_b={ave_b:.6f}\n")

    # 保存到cloud_k_b.json
    all_layers["model"] = "vit_large_patch16_384"
    if torch.cuda.is_available():
        all_layers["gpu"] = torch.cuda.get_device_name(0)
    
    # 从 src/profiler/ 向上两级到达项目根目录
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'assets', 'profiler_k_b')
    output_file = os.path.join(OUTPUT_DIR, 'cloud_k_b.json')
    
    with open(output_file, 'w') as f:
        json.dump(all_layers, f, indent=4)
    
    print(f"\nProfiler data saved to: {output_file}")
    print("Cloud profiler completed!")
