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
import csv
import math
import os
import sys

import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────────────────
# 将 src/ 加入 sys.path，使 schedule / utils / config 等模块可以直接 import
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import torch
from tqdm import tqdm

import config
from schedule.schedule import init, device_profiler, cloud_profiler, schedule
from schedule.declining_rate import declining_rate
from schedule.token_pruning import compute_token_schedule
from schedule.split_inference import device_forward, cloud_forward, _embed


# =====================================================================
# 第一部分：常量定义
# =====================================================================

# ── 6 个网络场景（与 Device-Only / Cloud-Only / Mixed 保持一致） ──────
NETWORK_SCENARIOS = [
    {"network_type": "LTE", "scenario": "static",  "trace_file": "static_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "walking", "trace_file": "walking_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "driving", "trace_file": "driving_lte_trace.csv"},
    {"network_type": "5G",  "scenario": "static",  "trace_file": "static_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "walking", "trace_file": "walking_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "driving", "trace_file": "driving_5g_trace.csv"},
]

METHOD = "janus"


# =====================================================================
# 第二部分：带宽读取
# =====================================================================

def load_bandwidth_series(trace_file):
    """
    从网络 trace CSV 中读取完整带宽序列（Mbps），过滤非法值。

    Args:
        trace_file: str, trace 文件的完整路径

    Returns:
        list[float]: 带宽序列（Mbps），已过滤掉 <= 0 的值

    Raises:
        ValueError: 如果过滤后序列为空
    """
    df = pd.read_csv(trace_file)
    series = [float(v) for v in df["bandwidth_mbps"] if float(v) > 0]
    if len(series) == 0:
        raise ValueError(f"Trace file {trace_file} 中无有效带宽值（全部 <= 0）")
    return series


def get_bandwidth_for_sample(bandwidth_series, sample_idx):
    """
    按样本索引循环映射到 trace 带宽序列中的某个时间步。
    trace 比样本少时自动循环；trace 比样本长时只用前面一段。

    Args:
        bandwidth_series: list[float], 带宽序列
        sample_idx:       int, 样本索引

    Returns:
        float: 该样本对应的带宽 (Mbps)
    """
    return bandwidth_series[sample_idx % len(bandwidth_series)]


# =====================================================================
# 第三部分：Janus 时延预测（任务 4-6）
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
# 第四部分：Janus 真实推理（任务 7-8）
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
# 第五部分：单样本评估（任务 14）
# =====================================================================

def run_janus_sample(model, image, label, sample_id,
                     N, x_0, D_M, bits, num_steps, step,
                     bandwidth_mbps, sla_ms, split_k):
    """
    对单个样本完成 Janus 完整评估流程：
      1. 取模型结构参数 N, x_0, D_M, bits
      2. 用当前带宽调用 scheduler
      3. 用 (alpha, split_layer) 预测 device_ms/cloud_ms/comm_ms/total_ms
      4. 用真实 split inference 算 pred_label
      5. 计算 correct / violate
      6. 返回结果 dict

    Args:
        model:          timm ViT 模型
        image:          Tensor (1, 3, 384, 384)
        label:          int, 真实标签
        sample_id:      int, 样本索引
        N:              int, Transformer 层数
        x_0:            int, 初始 token 数
        D_M:            int, token embedding 维度
        bits:           int, 数据类型占用 bit 数
        num_steps:      int, alpha 步数
        step:           float, alpha 步长
        bandwidth_mbps: float, 当前样本对应的带宽 (Mbps)
        sla_ms:         float, SLA 延迟上限（毫秒）
        split_k:        int, fine-to-coarse 候选点密度参数

    Returns:
        dict: 包含所有结果字段的记录
    """
    # 带宽转换：Mbps -> bps
    bandwidth_bps = bandwidth_mbps * 1e6

    # ── 1. 调用 scheduler ──
    alpha, split_layer = schedule(
        N, x_0, D_M, bits, num_steps, step, bandwidth_bps, sla_ms, split_k)

    # ── 2. 预测时延（profiler 线性模型） ──
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
        "bandwidth_mbps": bandwidth_mbps,
        "alpha": alpha,
        "split_layer": split_layer,
    }


# =====================================================================
# 第六部分：汇总与保存（任务 15-16）
# =====================================================================

def summarize(results, network_type, scenario):
    """
    计算汇总指标，包含场景元信息。格式与 baseline 完全一致。

    Returns:
        dict: 包含 network_type, scenario, method, num_samples,
              avg_device_ms, avg_cloud_ms, avg_comm_ms, avg_total_ms,
              accuracy, violation_ratio, throughput_fps
    """
    n = len(results)
    avg_device_ms = sum(r["device_ms"] for r in results) / n
    avg_cloud_ms = sum(r["cloud_ms"] for r in results) / n
    avg_comm_ms = sum(r["comm_ms"] for r in results) / n
    avg_total_ms = sum(r["total_time_ms"] for r in results) / n
    accuracy = sum(r["correct"] for r in results) / n
    violation_ratio = sum(r["violate"] for r in results) / n
    sum_total = sum(r["total_time_ms"] for r in results)
    throughput_fps = n / (sum_total / 1000.0) if sum_total > 0 else 0.0

    return {
        "network_type": network_type,
        "scenario": scenario,
        "method": METHOD,
        "num_samples": n,
        "avg_device_ms": avg_device_ms,
        "avg_cloud_ms": avg_cloud_ms,
        "avg_comm_ms": avg_comm_ms,
        "avg_total_ms": avg_total_ms,
        "accuracy": accuracy,
        "violation_ratio": violation_ratio,
        "throughput_fps": throughput_fps,
    }


# ── 保存逐样本明细到 CSV（比 baseline 多 alpha, split_layer 两列） ────
RECORD_COLUMNS = [
    "sample_id", "network_type", "scenario", "bandwidth_mbps", "method",
    "device_ms", "cloud_ms", "comm_ms", "total_time_ms",
    "correct", "pred_label", "true_label", "violate",
    "alpha", "split_layer",
]


def save_records_csv(results, network_type, scenario, output_path):
    """
    将结果保存为带场景标签的逐样本明细 CSV。
    每行的 bandwidth_mbps 从 results[i]["bandwidth_mbps"] 读取（逐样本动态带宽）。

    Args:
        results:        list[dict], 推理结果
        network_type:   str, "LTE" / "5G"
        scenario:       str, "static" / "walking" / "driving"
        output_path:    str, 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        for r in results:
            writer.writerow([
                r["sample_id"],
                network_type,
                scenario,
                f"{r['bandwidth_mbps']:.4f}",
                METHOD,
                f"{r['device_ms']:.4f}",
                f"{r['cloud_ms']:.4f}",
                f"{r['comm_ms']:.4f}",
                f"{r['total_time_ms']:.4f}",
                r["correct"],
                r["pred_label"],
                r["true_label"],
                r["violate"],
                f"{r['alpha']:.4f}",
                r["split_layer"],
            ])

    print(f"  [Save] records -> {output_path}")


# ── 保存汇总指标到 CSV（与 baseline 完全一致） ────────────────────────
SUMMARY_COLUMNS = [
    "network_type", "scenario", "method", "num_samples",
    "avg_device_ms", "avg_cloud_ms", "avg_comm_ms", "avg_total_ms",
    "accuracy", "violation_ratio", "throughput_fps",
]


def save_summary_csv(summary, output_path):
    """
    保存单个场景的汇总指标到独立 CSV 文件。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerow([
            summary["network_type"],
            summary["scenario"],
            summary["method"],
            summary["num_samples"],
            f"{summary['avg_device_ms']:.4f}",
            f"{summary['avg_cloud_ms']:.4f}",
            f"{summary['avg_comm_ms']:.4f}",
            f"{summary['avg_total_ms']:.4f}",
            f"{summary['accuracy']:.6f}",
            f"{summary['violation_ratio']:.6f}",
            f"{summary['throughput_fps']:.4f}",
        ])

    print(f"  [Save] summary -> {output_path}")


# =====================================================================
# 第七部分：主流程（任务 9-13）
# =====================================================================

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
    num_steps = int(a_max / step)

    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, D_M={D_M}, bits={bits}")
    print(f"[Config] SLA={args.sla}ms, split_k={args.split_k}")
    print(f"[Scheduler] alpha_max={a_max:.4f}, step={step}, num_steps={num_steps}")
    print(f"[Comm]   通信大小按中间 token 张量大小计算（非原图字节数）")

    # 1-3. 加载数据集
    from utils.imagenet_loader import get_imagenet_loader
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

        # 2-1. 读取该场景完整带宽 trace
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bw_series = load_bandwidth_series(trace_path)
        avg_bw = sum(bw_series) / len(bw_series)
        min_bw = min(bw_series)
        max_bw = max(bw_series)

        print(f"\n  [{tag}] trace_len={len(bw_series)}, "
              f"avg_bw={avg_bw:.2f}, min_bw={min_bw:.2f}, max_bw={max_bw:.2f} Mbps")

        # 2-2. 遍历整个数据集
        results = []
        split_counts = {}  # 统计 split_layer 分布

        for sample_idx, (images, labels) in enumerate(
                tqdm(loader, desc=f"Janus [{tag}]", total=total_samples)):
            images = images.to(dev)
            label = int(labels.item())

            # 2-3. 当前样本取当前带宽
            bandwidth_mbps = get_bandwidth_for_sample(bw_series, sample_idx)

            # 2-4. 调用 Janus 单样本评估
            record = run_janus_sample(
                model, images, label, sample_idx,
                N, x_0, D_M, bits, num_steps, step,
                bandwidth_mbps, args.sla, args.split_k)

            results.append(record)

            # 统计 split_layer 分布
            sl = record["split_layer"]
            split_counts[sl] = split_counts.get(sl, 0) + 1

        # 2-5. 场景跑完：保存 records
        records_path = os.path.join(output_dir, f"janus_{tag}_records.csv")
        save_records_csv(results, net_type, scenario, records_path)

        # 2-6. 计算并保存 summary
        summary = summarize(results, net_type, scenario)
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
