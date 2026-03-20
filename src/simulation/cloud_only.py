"""
Cloud-Only Baseline — Fig.7 结果生成器
=======================================
功能：
  - 固定每层裁剪 23 tokens 的 baseline pruning（论文 Fig.7 口径）
  - split_layer = 0（全部在 cloud 端执行，device_ms = 0）
  - 使用 cloud profiler 线性模型预测 cloud 端逐层延迟（不使用真实计时）
  - comm_ms = 初始 token 张量大小 (x_0 * D_M * bits) / 动态带宽
    与 Janus split_layer=0 完全一致，传输的是 embedding 后的 token tensor
  - 使用真实 forward 得到分类结果计算准确率（本地执行，语义上代表云端）
  - 准确率只算一次复用；时延按 6 个网络场景分别计算，
    每个样本按 trace 时间步循环映射带宽，逐样本独立计算 comm_ms

输出目录（默认 results/cloud_only/）：
  每个场景生成两个文件：
    cloud_only_{scenario}_records.csv   — 逐样本明细
    cloud_only_{scenario}_summary.csv   — 汇总指标

用法：
  python src/simulation/cloud_only.py [--data_path DATA] [--sla SLA_MS]
                                      [--prune_per_layer 23] [--output_dir DIR]
"""

import argparse
import csv
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
from schedule.schedule import init, cloud_profiler
from schedule.token_pruning import prune_tokens
from schedule.split_inference import _embed
from utils.imagenet_loader import get_imagenet_loader


# ── 6 个网络场景（结构化，与 Device-Only / Mixed / Janus 保持一致） ────
NETWORK_SCENARIOS = [
    {"network_type": "LTE", "scenario": "static",  "trace_file": "static_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "walking", "trace_file": "walking_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "driving", "trace_file": "driving_lte_trace.csv"},
    {"network_type": "5G",  "scenario": "static",  "trace_file": "static_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "walking", "trace_file": "walking_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "driving", "trace_file": "driving_5g_trace.csv"},
]

METHOD = "cloud_only"


# ── 固定每层裁剪 baseline 的 token schedule ──────────────────────────
def build_fixed_baseline_schedule(N, x_0, prune_per_layer=23):
    """
    构建固定每层裁剪 prune_per_layer 个 token 的 baseline pruning 计划。
    论文 Fig.7：ViT-L baseline = 每层固定裁剪 23 tokens。

    Args:
        N:               int, Transformer 层数
        x_0:             int, 初始 token 数（含 CLS）
        prune_per_layer: int, 每层裁剪的 token 数

    Returns:
        dict: {l: keep_n}  l = 1..N，每层经过 pruning 后保留的 token 数
    """
    x_l = {}
    remaining = x_0
    for l in range(1, N + 1):
        remaining = max(remaining - prune_per_layer, 1)
        x_l[l] = remaining
    return x_l


# ── 预测 cloud-only 推理时间（cloud profiler 线性模型） ────────────────
def predict_cloud_time(token_schedule, N):
    """
    使用 cloud profiler 线性模型预测 cloud-only 整体推理时间。
    时间 = Σ_{l=1}^{N} cloud_profiler(x_l[l], l)

    Args:
        token_schedule: dict, {l: keep_n}
        N:              int,  Transformer 层数

    Returns:
        cloud_ms: float, 预测总延迟（毫秒）
    """
    total_ms = 0.0
    for l in range(1, N + 1):
        total_ms += cloud_profiler(token_schedule[l], l)
    return total_ms


# ── 读取完整带宽序列 ──────────────────────────────────────────────────
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


# ── 按样本索引映射带宽（循环复用） ────────────────────────────────────
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


# ── 按初始 token 张量大小计算通信时间（与 Janus split_layer=0 一致） ──
def predict_tensor_transfer_time(x_0, D_M, bits, bandwidth_mbps):
    """
    计算 cloud-only 路径的通信时间：传输 embedding 后的 token tensor。
    comm_ms = x_0 * D_M * bits / (bandwidth_mbps * 1e6) * 1000

    与 Janus scheduler 中 split_layer=0 的通信量计算方式完全一致，
    确保 Cloud-Only baseline 与 Janus cloud-only 退化路径口径统一。

    Args:
        x_0:            int, 初始 token 数（含 CLS）
        D_M:            int, token embedding 维度
        bits:           int, 数据类型占用 bit 数
        bandwidth_mbps: float, 当前带宽 (Mbps)

    Returns:
        comm_ms: float, 传输时间（毫秒）

    Raises:
        ValueError: 如果 bandwidth_mbps <= 0
    """
    if bandwidth_mbps <= 0:
        raise ValueError(f"带宽必须 > 0，当前值: {bandwidth_mbps}")
    transferred_bits = x_0 * D_M * bits
    comm_ms = transferred_bits / (bandwidth_mbps * 1e6) * 1000
    return comm_ms


# ── 实际推理（用于准确率计算，语义上代表云端完整执行） ──────────────────
def run_cloud_only_inference(model, image, token_schedule, N):
    """
    逻辑上表示云端完整执行所有 N 层 + norm + head，
    按 token_schedule 进行固定 pruning。
    实际在本地执行 forward，仅用于得到准确率。

    Args:
        model:          timm ViT 模型
        image:          Tensor (1, 3, 384, 384)
        token_schedule: dict, {l: keep_n}
        N:              int, 层数

    Returns:
        pred_label: int, 预测类别
        logits:     Tensor
    """
    x = _embed(model, image)

    for l in range(1, N + 1):
        x = model.blocks[l - 1](x)
        x = prune_tokens(x, token_schedule[l])

    # norm + classification head（CLS token）
    x = model.norm(x)
    logits = model.head(x[:, 0])
    pred_label = int(logits.argmax(dim=-1).item())
    return pred_label, logits


# ── 汇总指标 ──────────────────────────────────────────────────────────
def summarize(results, network_type, scenario):
    """
    计算汇总指标，包含场景元信息。

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


# ── 保存逐样本明细到 CSV ──────────────────────────────────────────────
RECORD_COLUMNS = [
    "sample_id", "network_type", "scenario", "bandwidth_mbps", "method",
    "device_ms", "cloud_ms", "comm_ms", "total_time_ms",
    "correct", "pred_label", "true_label", "violate",
]

def save_records_csv(results, network_type, scenario, output_path):
    """
    将结果保存为带场景标签的逐样本明细 CSV。
    每行的 bandwidth_mbps 从 results[i]["bandwidth_mbps"] 读取（逐样本动态带宽）。

    Args:
        results:        list[dict], 推理结果（含该场景的时间字段及逐样本带宽）
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
            ])

    print(f"  [Save] records → {output_path}")


# ── 保存汇总指标到 CSV ────────────────────────────────────────────────
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

    print(f"  [Save] summary → {output_path}")


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cloud-Only Baseline — Fig.7 Result Generator")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH),
                        help="ImageNet parquet 文件路径")
    parser.add_argument("--sla", type=float, default=config.DEFAULT_SLA,
                        help="SLA 延迟上限（毫秒），默认 300ms")
    parser.add_argument("--prune_per_layer", type=int, default=23,
                        help="每层固定裁剪 token 数（论文 Fig.7 baseline = 23）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/cloud_only/）")
    args = parser.parse_args()

    # ================================================================
    # Phase 1: 执行一次 forward 推理，缓存准确率相关结果
    # ================================================================
    print("=" * 60)
    print("Cloud-Only Baseline — Fig.7 Result Generator")
    print("=" * 60)

    # 1-1. 加载模型
    model = init()
    dev = next(model.parameters()).device
    N = len(model.blocks)           # 24
    x_0 = model.pos_embed.size(1)   # 577
    D_M = model.pos_embed.size(2)   # 1024
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits  # 32

    # 1-2. 构建固定 baseline pruning 计划
    token_schedule = build_fixed_baseline_schedule(N, x_0, args.prune_per_layer)
    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, device={dev}")
    print(f"[Config] prune_per_layer={args.prune_per_layer}, SLA={args.sla}ms")
    print(f"[Schedule] layer 1 keep={token_schedule[1]}, "
          f"layer {N} keep={token_schedule[N]}")
    print(f"[Comm]   通信大小按 token tensor 计算: x_0={x_0} * D_M={D_M} * bits={bits} = {x_0*D_M*bits} bits")

    # 1-3. 加载数据集
    loader = get_imagenet_loader(args.data_path, batch_size=1, num_workers=0)
    total_samples = len(loader.dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # 1-4. 逐样本推理（只跑一次）
    #       cloud_ms 固定，comm_ms 由 token tensor 大小和动态带宽共同决定，留到 Phase 2
    cloud_ms_value = predict_cloud_time(token_schedule, N)
    print(f"[Profiler] cloud_ms (all samples) = {cloud_ms_value:.4f} ms")

    inference_cache = []
    for sample_idx, (images, labels) in enumerate(
            tqdm(loader, desc="Cloud-Only Inference", total=total_samples)):
        images = images.to(dev)
        label = int(labels.item())

        with torch.no_grad():
            pred_label, _ = run_cloud_only_inference(
                model, images, token_schedule, N)

        correct = 1 if pred_label == label else 0
        inference_cache.append({
            "sample_id": sample_idx,
            "pred_label": pred_label,
            "true_label": label,
            "correct": correct,
            "cloud_ms": cloud_ms_value,
        })

    # ── 控制台输出准确率概览 ──
    total_correct = sum(r["correct"] for r in inference_cache)
    accuracy = total_correct / total_samples
    print(f"\n===== Cloud-Only Inference Done =====")
    print(f"  Accuracy             : {accuracy:.6f}")
    print(f"  Cloud Time (ms)      : {cloud_ms_value:.4f}")
    print("=" * 60)

    # ================================================================
    # Phase 2: 对每个网络场景，重新计算 comm_ms / total_time_ms / violate
    #           分别保存 records + summary
    # ================================================================
    if args.output_dir is None:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "cloud_only")
    else:
        output_dir = args.output_dir

    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 按 {len(NETWORK_SCENARIOS)} 个网络场景分别计算时延 ...")
    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}"

        # 读取该场景完整带宽序列
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bw_series = load_bandwidth_series(trace_path)
        avg_bw = sum(bw_series) / len(bw_series)
        min_bw = min(bw_series)
        max_bw = max(bw_series)

        print(f"\n  [{tag}] trace_len={len(bw_series)}, "
              f"avg_bw={avg_bw:.2f}, min_bw={min_bw:.2f}, max_bw={max_bw:.2f} Mbps")

        # 为该场景逐样本计算 comm_ms（动态带宽 + token tensor 大小）
        scenario_results = []
        for cached in inference_cache:
            sid = cached["sample_id"]
            bw = get_bandwidth_for_sample(bw_series, sid)
            comm_ms = predict_tensor_transfer_time(x_0, D_M, bits, bw)
            total_time_ms = cached["cloud_ms"] + comm_ms
            violate = 1 if total_time_ms > args.sla else 0

            scenario_results.append({
                "sample_id": sid,
                "device_ms": 0.0,
                "cloud_ms": cached["cloud_ms"],
                "comm_ms": comm_ms,
                "total_time_ms": total_time_ms,
                "correct": cached["correct"],
                "pred_label": cached["pred_label"],
                "true_label": cached["true_label"],
                "violate": violate,
                "bandwidth_mbps": bw,
            })

        # 保存逐样本明细
        records_path = os.path.join(output_dir, f"cloud_only_{tag}_records.csv")
        save_records_csv(scenario_results, net_type, scenario, records_path)

        # 计算并保存汇总
        summary = summarize(scenario_results, net_type, scenario)
        summary_path = os.path.join(output_dir, f"cloud_only_{tag}_summary.csv")
        save_summary_csv(summary, summary_path)

        # 控制台输出该场景概览
        print(f"    Avg Comm (ms)  : {summary['avg_comm_ms']:.4f}")
        print(f"    Avg Total (ms) : {summary['avg_total_ms']:.4f}")
        print(f"    Violation Ratio: {summary['violation_ratio']:.6f}")
        print(f"    Throughput (fps): {summary['throughput_fps']:.4f}")

    print(f"\n[Done] 共保存 {len(NETWORK_SCENARIOS) * 2} 个文件到 {output_dir}")


if __name__ == "__main__":
    main()
