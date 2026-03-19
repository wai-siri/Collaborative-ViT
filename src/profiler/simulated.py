"""
模拟 Profiler 数据生成脚本
=========================
为 ViT-L@384（24 层）生成模拟的 device 端 k, b 参数，
并覆盖写入 assets/profiler_k_b/device_k_b.json。

每层的 k 在 0.0475 左右、b 在 -0.3 左右，
加入微小随机扰动使各层略有差异，更贴近真实场景。

用法:
    python src/profiler/simulated.py
"""

import json
import os
import random

# ──────────────────────── 配置 ────────────────────────
NUM_BLOCKS = 24
K_CENTER = 0.00255       # k 的中心值
B_CENTER = 0.02         # b 的中心值
K_JITTER = 0.00005        # k 的随机扰动幅度 (±)
B_JITTER = 0.005         # b 的随机扰动幅度 (±)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, '..', '..', 'assets', 'profiler_k_b', 'cloud_k_b.json')


def generate():
    """生成 24 层模拟 k, b 数据并写入 JSON。"""
    random.seed(42)  # 固定种子，保证可复现

    data = {}
    for i in range(NUM_BLOCKS):
        k = K_CENTER + random.uniform(-K_JITTER, K_JITTER)
        b = B_CENTER + random.uniform(-B_JITTER, B_JITTER)
        data[str(i)] = {"k": round(k, 6), "b": round(b, 4)}

    data["model"] = "vit_large_patch16_384"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"已生成模拟数据并保存到 {OUTPUT_PATH}")
    print(f"共 {NUM_BLOCKS} 层，k ≈ {K_CENTER}, b ≈ {B_CENTER}")

    # 预览前 3 层
    for i in range(min(3, NUM_BLOCKS)):
        entry = data[str(i)]
        print(f"  block {i}: k={entry['k']}, b={entry['b']}")


if __name__ == "__main__":
    generate()
