"""
Janus Lightweight Profiler
==========================
对 ViT-L@384 的每个 Transformer block 进行逐层延迟采样，
拟合线性模型 T_l = k * tokens + b，输出 JSON 供调度器使用。

用法:
    python test_time.py <output_filename>
    例如:
        python test_time.py edge_k_b.json
        python test_time.py cloud_k_b.json
        python test_time.py 3090_k_b.json

输出文件保存在 assets/profiler_k_b/ 目录下。
"""

import argparse
import json
import time
import os

import numpy as np
import timm
import torch

# ──────────────────────────── 全局配置 ────────────────────────────
# 固定采样 token 点（ViT-L@384，patch 16 → 577 tokens）
TOKEN_SAMPLE_POINTS = [577, 512, 448, 384, 320, 256, 192, 128, 96, 64]

WARMUP_ITERS = 10    # 每个 (block, token_count) 的预热次数
MEASURE_ITERS = 30   # 每个 (block, token_count) 的正式测量次数

# ──────────────────────────── 模型加载 ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_large_patch16_384', pretrained=False)
model.eval()
model = model.to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────── 核心函数 ────────────────────────────
def warm_up(block, tokens):
    """对指定 block 和 token 输入执行预热，排除冷启动影响。"""
    for _ in range(WARMUP_ITERS):
        block(tokens)
    torch.cuda.synchronize()


def measure_single_point(block, tokens):
    """
    对单个 (block, token_count) 执行多次测量，返回 median latency（毫秒）。
    使用 median 而非 mean，可抵抗偶发尖峰干扰。
    """
    latencies = []
    for _ in range(MEASURE_ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        block(tokens)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # 转为毫秒
    return float(np.median(latencies))


def profile_block(block_idx, dim):
    """
    对第 block_idx 层，遍历所有采样 token 点，
    每个点先预热再测量，返回 (token_counts, latencies) 两个数组。
    """
    block = model.blocks[block_idx]
    token_counts = []
    latencies = []

    with torch.no_grad():
        for num_tokens in TOKEN_SAMPLE_POINTS:
            tokens = torch.randn(1, num_tokens, dim, device=device)

            # 预热：每个 (block, token_count) 都预热
            warm_up(block, tokens)

            # 正式测量
            lat = measure_single_point(block, tokens)
            token_counts.append(num_tokens)
            latencies.append(lat)
            print(f"  token={num_tokens:>4d}  latency={lat:.4f} ms")

    return np.array(token_counts), np.array(latencies)


# ──────────────────────────── 主流程 ────────────────────────────
if __name__ == "__main__":
    # 解析命令行参数：输出文件名
    parser = argparse.ArgumentParser(description="Janus Lightweight Profiler")
    parser.add_argument("output", type=str,
                        help="输出文件名，例如 edge_k_b.json / cloud_k_b.json")
    args = parser.parse_args()

    N = len(model.blocks)
    dim = model.pos_embed.size(2)  # 隐藏层维度

    all_layers = {}

    for block_idx in range(N):
        print(f"[block {block_idx}/{N-1}]")

        # 采样 + 测量
        token_counts, latencies = profile_block(block_idx, dim)

        # 一次 polyfit，得到该层最终 k, b
        k, b = np.polyfit(token_counts, latencies, deg=1)
        all_layers[str(block_idx)] = {"k": float(k), "b": float(b)}
        print(f"  => k={k:.6f}, b={b:.4f}\n")

    # 写入模型名称
    all_layers["model"] = "vit_large_patch16_384"

    # 保存 JSON
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', '..', 'assets', 'profiler_k_b')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, args.output)
    with open(output_path, 'w') as f:
        json.dump(all_layers, f, indent=4)
    print(f"结果已保存到 {output_path}")
