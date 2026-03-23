"""
Janus — Fig.7 结果生成器
========================
功能：
  - 对每个样本 / 每个网络条件，调用 Janus 动态调度器选择最优 (alpha, split_layer)
  - 使用 profiler 线性模型预测 device / cloud 端逐层延迟（与 baseline 同口径）
  - comm_ms = 中间 token 张量大小 / 动态带宽（传输 split 后的中间表示，非原图）
  - 使用真实 forward（device_forward + cloud_forward）得到分类结果计算准确率
  - 逐网络场景、逐样本独立调度并执行推理

  split_layer 语义约定：
    - 0:     cloud-only，device 不执行 transformer layer
    - 1..N:  真实 split inference
    - N+1:   device-only，cloud 不执行 transformer layer

输出目录（默认 results/janus/）：
  每个场景生成两个文件：
    janus_{scenario}_records.csv   — 逐样本明细（比 baseline 多 alpha, split_layer 列）
    janus_{scenario}_summary.csv   — 汇总指标（与 baseline 完全一致）

用法：
  python src/simulation/janus.py [--data_path DATA] [--sla 300]
                                  [--split_k 5] [--output_dir DIR]
"""

import argparse
import os
import sys

# ── 路径设置 ──────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import torch
from tqdm import tqdm

import config
from schedule.schedule import init, device_profiler, cloud_profiler, schedule
from schedule.declining_rate import declining_rate
from schedule.token_pruning import compute_token_schedule
from schedule.split_inference import device_forward, cloud_forward
from utils.imagenet_loader import get_imagenet_loader
from simulation.baseline_common import (
    NETWORK_SCENARIOS, load_bandwidth_series, get_bandwidth_for_sample,
    estimate_bandwidth, summarize, save_records_csv, save_summary_csv,
)

METHOD = "janus"


# =====================================================================
# Janus 特有：时延预测
# =====================================================================

def predict_janus_latency(N, x_0, D_M, bits, alpha, split_layer, bandwidth_bps):
    """
    Janus 时延分解函数 — 基于 profiler 线性模型预测，不做真实 forward。

    时间口径与 baseline 完全一致：
      - device_ms:  累加 layer 1..split_layer 的 device profiler
      - cloud_ms:   累加 layer (split_layer+1)..N 的 cloud profiler
      - comm_ms:    中间 token 张量大小 / 带宽（非原图字节数）

    split_layer 语义：
      - 0:     cloud-only -> device_ms=0, comm_ms 传输 x_0 * D_M * bits
      - 1..N:  真实 split  -> comm_ms 传输 x_l[split_layer] * D_M * bits
      - N+1:   device-only -> cloud_ms=0, comm_ms=0

    token 数来自 compute_token_schedule(alpha, N, x_0)，保证与 scheduler
    和真实 forward 使用完全相同的 token 逻辑。

    Args:
        N:              int, Transformer 层数
        x_0:            int, 初始 token 数（含 CLS）
        D_M:            int, token embedding 维度
        bits:           int, 数据类型占用 bit 数
        alpha:          float, declining rate
        split_layer:    int, split point (0 to N+1)
        bandwidth_bps:  float, 当前带宽 (bps)

    Returns:
        dict: {device_ms, cloud_ms, comm_ms, total_ms}
    """
    # 使用与 scheduler / 真实 forward 完全相同的 token schedule
    x_l = compute_token_schedule(alpha, N, x_0)

    # ── device_ms ──
    device_ms = 0.0
    if split_layer >= 1:
        for l in range(1, min(split_layer, N) + 1):
            device_ms += device_profiler(x_l[l], l)

    # ── cloud_ms ──
    cloud_ms = 0.0
    if split_layer <= N:
        for l in range(split_layer + 1, N + 1):
            cloud_ms += cloud_profiler(x_l[l], l)

    # ── comm_ms：中间 token 张量大小 / 带宽 ──
    if split_layer >= N + 1:
        # device-only: 无通信
        comm_ms = 0.0
    elif split_layer == 0:
        # cloud-only: 传输初始 token 表示 (embedding 后)
        transferred_bits = x_0 * D_M * bits
        comm_ms = transferred_bits / bandwidth_bps * 1000
    else:
        # 真实 split: 传输 split 后的中间 token 表示
        transferred_bits = x_l[split_layer] * D_M * bits
        comm_ms = transferred_bits / bandwidth_bps * 1000

    total_ms = device_ms + cloud_ms + comm_ms

    return {
        "device_ms": device_ms,
        "cloud_ms": cloud_ms,
        "comm_ms": comm_ms,
        "total_ms": total_ms,
    }


# =====================================================================
# Janus 特有：真实推理
# =====================================================================

def run_janus_inference(model, image, alpha, split_layer):
    """
    执行一次完整的 Janus split inference，覆盖三种边界路径。

    - split_layer = 0:     device 只做 embedding -> cloud 执行全部 N 层
    - 1 <= split_layer <= N: device 跑前半段 -> cloud 跑后半段
    - split_layer = N+1:   device 执行全部 N 层 -> cloud 只做 norm + head

    device 和 cloud 使用同一个 alpha，保证 pruning 贯穿整条推理链。

    Args:
        model:       timm ViT 模型
        image:       Tensor (1, 3, 384, 384)
        alpha:       float, declining rate
        split_layer: int, split point (0 to N+1)

    Returns:
        pred_label: int, 预测类别
    """
    with torch.no_grad():
        # device 侧：embedding + layer 1..split_layer
        x_mid = device_forward(model, image, alpha, split_layer)
        # cloud 侧：layer (split_layer+1)..N + norm + head
        logits = cloud_forward(model, x_mid, split_layer, alpha)

    pred_label = int(logits.argmax(dim=-1).item())
    return pred_label


# =====================================================================
# Janus 特有：单样本评估
# =====================================================================

def run_janus_sample(model, image, label, sample_id,
                     N, x_0, D_M, bits, num_steps, step,
                     observed_bandwidth_mbps, estimated_bandwidth_mbps,
                     sla_ms, split_k):
    """
    对单个样本完成 Janus 完整评估流程：
      1. 取模型结构参数 N, x_0, D_M, bits
      2. 用估计带宽调用 scheduler
      3. 用 (alpha, split_layer) 预测 device_ms/cloud_ms/comm_ms/total_ms
      4. 用真实 split inference 算 pred_label
      5. 计算 correct / violate
      6. 返回结果 dict

    Args:
        model:                     timm ViT 模型
        image:                     Tensor (1, 3, 384, 384)
        label:                     int, 真实标签
        sample_id:                 int, 样本索引
        N:                         int, Transformer 层数
        x_0:                       int, 初始 token 数
        D_M:                       int, token embedding 维度
        bits:                      int, 数据类型占用 bit 数
        num_steps:                 int, alpha 步数
        step:                      float, alpha 步长
        observed_bandwidth_mbps:   float, 当前 trace 时间步的真实带宽 (Mbps)，仅记录
        estimated_bandwidth_mbps:  float, 调和平均估计带宽 (Mbps)，用于 scheduler 和 comm_ms
        sla_ms:                    float, SLA 延迟上限（毫秒）
        split_k:                   int, fine-to-coarse 候选点密度参数

    Returns:
        dict: 包含所有结果字段的记录
    """
    # 估计带宽转换：Mbps -> bps
    bandwidth_bps = estimated_bandwidth_mbps * 1e6

    # ── 1. 调用 scheduler（基于估计带宽） ──
    alpha, split_layer = schedule(
        N, x_0, D_M, bits, num_steps, step, bandwidth_bps, sla_ms, split_k)

    # ── 2. 预测时延（profiler 线性模型，基于估计带宽） ──
    latency = predict_janus_latency(
        N, x_0, D_M, bits, alpha, split_layer, bandwidth_bps)

    # ── 3. 真实推理（用于准确率） ──
    pred_label = run_janus_inference(model, image, alpha, split_layer)

    # ── 4. 计算 correct / violate ──
    correct = 1 if pred_label == label else 0
    violate = 1 if latency["total_ms"] > sla_ms else 0

    return {
        "sample_id": sample_id,
        "device_ms": latency["device_ms"],
        "cloud_ms": latency["cloud_ms"],
        "comm_ms": latency["comm_ms"],
        "total_time_ms": latency["total_ms"],
        "correct": correct,
        "pred_label": pred_label,
        "true_label": label,
        "violate": violate,
        "observed_bandwidth_mbps": observed_bandwidth_mbps,
        "estimated_bandwidth_mbps": estimated_bandwidth_mbps,
        "alpha": alpha,
        "split_layer": split_layer,
    }


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Janus — Fig.7 Result Generator (dynamic pruning + splitting)")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH),
                        help="ImageNet parquet 文件路径")
    parser.add_argument("--sla", type=float, default=config.DEFAULT_SLA,
                        help="SLA 延迟上限（毫秒），默认 300ms")
    parser.add_argument("--split_k", type=int, default=5,
                        help="fine-to-coarse 候选点密度参数（论文参数 k），默认 5")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/janus/）")
    args = parser.parse_args()

    # ================================================================
    # Phase 1: 初始化
    # ================================================================
    print("=" * 60)
    print("Janus — Fig.7 Result Generator")
    print("=" * 60)

    # 1-1. 加载模型
    model = init()
    dev = next(model.parameters()).device
    N = len(model.blocks)           # 24
    x_0 = model.pos_embed.size(1)   # 577
    D_M = model.pos_embed.size(2)   # 1024
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits  # 32

    # 1-2. 计算 alpha_max 和步长
    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step) + 1

    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, D_M={D_M}, bits={bits}")
    print(f"[Config] SLA={args.sla}ms, split_k={args.split_k}")
    print(f"[Scheduler] alpha_max={a_max:.4f}, step={step}, num_steps={num_steps}")
    print(f"[Comm]   通信大小按中间 token 张量大小计算（非原图字节数）")

    # 1-3. 加载数据集
    loader = get_imagenet_loader(args.data_path, batch_size=1, num_workers=0)
    total_samples = len(loader.dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # ================================================================
    # Phase 2: 逐网络场景、逐样本调度并推理
    # ================================================================
    if args.output_dir is None:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "janus")
    else:
        output_dir = args.output_dir

    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 逐网络场景遍历数据集，每样本独立调度 ...")
    print(f"  共 {len(NETWORK_SCENARIOS)} 个场景 x {total_samples} 个样本")

    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}"
        cold_start_bw = sc["cold_start_bw"]

        # 2-1. 读取该场景完整带宽 trace
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bw_series = load_bandwidth_series(trace_path, bw_floor=sc["bw_floor"])
        avg_bw = sum(bw_series) / len(bw_series)
        min_bw = min(bw_series)
        max_bw = max(bw_series)

        print(f"\n  [{tag}] trace_len={len(bw_series)}, "
              f"avg_bw={avg_bw:.2f}, min_bw={min_bw:.2f}, max_bw={max_bw:.2f} Mbps, "
              f"cold_start_bw={cold_start_bw} Mbps")

        # 2-2. 遍历整个数据集
        results = []
        split_counts = {}  # 统计 split_layer 分布

        for sample_idx, (images, labels) in enumerate(
                tqdm(loader, desc=f"Janus [{tag}]", total=total_samples)):
            images = images.to(dev)
            label = int(labels.item())

            # 2-3. 当前样本的观测带宽和估计带宽
            observed_bw = get_bandwidth_for_sample(bw_series, sample_idx)
            estimated_bw = estimate_bandwidth(bw_series, sample_idx, cold_start_bw)

            # 2-4. 调用 Janus 单样本评估（基于估计带宽）
            record = run_janus_sample(
                model, images, label, sample_idx,
                N, x_0, D_M, bits, num_steps, step,
                observed_bw, estimated_bw, args.sla, args.split_k)

            results.append(record)

            # 统计 split_layer 分布
            sl = record["split_layer"]
            split_counts[sl] = split_counts.get(sl, 0) + 1

        # 2-5. 场景跑完：保存 records（Janus 多 alpha, split_layer 两列）
        records_path = os.path.join(output_dir, f"janus_{tag}_records.csv")
        save_records_csv(results, METHOD, net_type, scenario, records_path,
                         extra_columns=["alpha", "split_layer"])

        # 2-6. 计算并保存 summary
        summary = summarize(results, METHOD, net_type, scenario)
        summary_path = os.path.join(output_dir, f"janus_{tag}_summary.csv")
        save_summary_csv(summary, summary_path)

        # 2-7. 控制台输出该场景概览
        avg_alpha = sum(r["alpha"] for r in results) / len(results)
        print(f"    Avg Alpha      : {avg_alpha:.4f}")
        print(f"    Split Dist     : {dict(sorted(split_counts.items()))}")
        print(f"    Avg Device (ms): {summary['avg_device_ms']:.4f}")
        print(f"    Avg Cloud (ms) : {summary['avg_cloud_ms']:.4f}")
        print(f"    Avg Comm (ms)  : {summary['avg_comm_ms']:.4f}")
        print(f"    Avg Total (ms) : {summary['avg_total_ms']:.4f}")
        print(f"    Accuracy       : {summary['accuracy']:.6f}")
        print(f"    Violation Ratio: {summary['violation_ratio']:.6f}")
        print(f"    Throughput (fps): {summary['throughput_fps']:.4f}")

    print(f"\n[Done] 共保存 {len(NETWORK_SCENARIOS) * 2} 个文件到 {output_dir}")


if __name__ == "__main__":
    main()
