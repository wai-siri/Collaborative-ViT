"""
Simulation 公共模块 — 供 device_only / cloud_only / mixed / janus 复用
=====================================================================
包含：
  - NETWORK_SCENARIOS:        6 个网络场景定义
  - RECORD_COLUMNS:           baseline 逐样本 CSV 列定义
  - SUMMARY_COLUMNS:          汇总 CSV 列定义
  - build_fixed_baseline_schedule: 固定每层裁剪的 token schedule
  - predict_device_time:      device profiler 线性模型预测
  - predict_cloud_time:       cloud profiler 线性模型预测
  - predict_tensor_transfer_time: token tensor 传输时间预测
  - load_bandwidth_series:    从 trace CSV 读取完整带宽序列
  - get_bandwidth_for_sample: 按样本索引循环映射带宽（所有脚本统一使用此方式记录逐样本带宽）
  - estimate_bandwidth:       基于历史观测的调和平均带宽估计（冷启动 + harmonic mean）
  - run_pruned_vit_inference: 统一的 pruned ViT 推理（用于准确率）
  - summarize:                汇总指标计算
  - save_records_csv:         保存逐样本明细 CSV
  - save_summary_csv:         保存汇总指标 CSV
"""

import csv
import os

import pandas as pd
import torch

from schedule.schedule import device_profiler, cloud_profiler
from schedule.token_pruning import prune_tokens
from schedule.split_inference import _embed


# ── 6 个网络场景（LTE/5G × static/walking/driving） ──────────────────
# bw_floor: 仅用于数值保护（防止 trace 中接近 0 的值导致除零），统一用很小正数
# cold_start_bw: 论文中冷启动带宽估计值（Mbps），sample_idx==0 时使用
#   LTE cold_start_bw = 7.6 Mbps, 5G cold_start_bw = 14.7 Mbps
NETWORK_SCENARIOS = [
    {"network_type": "LTE", "scenario": "static",  "trace_file": "static_lte_trace.csv",  "bw_floor": 0.01, "cold_start_bw": 7.6},
    {"network_type": "LTE", "scenario": "walking", "trace_file": "walking_lte_trace.csv", "bw_floor": 0.01, "cold_start_bw": 7.6},
    {"network_type": "LTE", "scenario": "driving", "trace_file": "driving_lte_trace.csv", "bw_floor": 0.01, "cold_start_bw": 7.6},
    {"network_type": "5G",  "scenario": "static",  "trace_file": "static_5g_trace.csv",   "bw_floor": 0.01, "cold_start_bw": 14.7},
    {"network_type": "5G",  "scenario": "walking", "trace_file": "walking_5g_trace.csv",  "bw_floor": 0.01, "cold_start_bw": 14.7},
    {"network_type": "5G",  "scenario": "driving", "trace_file": "driving_5g_trace.csv",  "bw_floor": 0.01, "cold_start_bw": 14.7},
]


# ── baseline 逐样本 CSV 列定义 ───────────────────────────────────────
# observed_bandwidth_mbps: 当前 trace 时间步的真实带宽点值
# estimated_bandwidth_mbps: 基于调和平均的带宽估计值（用于计算 comm_ms）
RECORD_COLUMNS = [
    "sample_id", "network_type", "scenario",
    "observed_bandwidth_mbps", "estimated_bandwidth_mbps", "method",
    "device_ms", "cloud_ms", "comm_ms", "total_time_ms",
    "correct", "pred_label", "true_label", "violate",
]

# ── 汇总 CSV 列定义 ─────────────────────────────────────────────────
SUMMARY_COLUMNS = [
    "network_type", "scenario", "method", "num_samples",
    "avg_device_ms", "avg_cloud_ms", "avg_comm_ms", "avg_total_ms",
    "accuracy", "violation_ratio", "throughput_fps",
]


# =====================================================================
# Token Schedule
# =====================================================================

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


# =====================================================================
# 时延预测（Profiler 线性模型）
# =====================================================================

def predict_device_time(token_schedule, N):
    """
    使用 device profiler 线性模型预测 device-only 整体推理时间。
    时间 = Σ_{l=1}^{N} device_profiler(x_l[l], l)

    Args:
        token_schedule: dict, {l: keep_n}
        N:              int,  Transformer 层数

    Returns:
        device_ms: float, 预测总延迟（毫秒）
    """
    total_ms = 0.0
    for l in range(1, N + 1):
        total_ms += device_profiler(token_schedule[l], l)
    return total_ms


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


def predict_tensor_transfer_time(x_0, D_M, bits, bandwidth_mbps):
    """
    计算 cloud 路径的通信时间：传输 embedding 后的 token tensor。
    comm_ms = x_0 * D_M * bits / (bandwidth_mbps * 1e6) * 1000

    与 Janus scheduler 中 split_layer=0 的通信量计算方式完全一致，
    确保 baseline 与 Janus cloud-only 退化路径口径统一。

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


# =====================================================================
# 带宽读取
# =====================================================================

def load_bandwidth_series(trace_file, bw_floor=1.0):
    """
    从网络 trace CSV 中读取完整带宽序列（Mbps），并对极小值做 floor clipping。

    trace 中可能存在接近 0 的极小正值（如 0.003 Mbps），这些是网络中断期间的
    残余噪声，如果直接参与 comm_ms 计算会导致通信时间爆炸（数百万 ms）。
    因此将所有低于 bw_floor 的值统一抬到 bw_floor，保持 trace 原始长度不变。

    Args:
        trace_file: str, trace 文件的完整路径
        bw_floor:   float, 带宽下限（Mbps），低于此值的统一抬到此值
                    典型值：LTE=1.0, 5G=5.0

    Returns:
        list[float]: 带宽序列（Mbps），长度与原始 trace 一致，最小值 = bw_floor
    """
    df = pd.read_csv(trace_file)
    series = [max(float(v), bw_floor) for v in df["bandwidth_mbps"]]
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


def estimate_bandwidth(bandwidth_series, sample_idx, cold_start_bw):
    """
    带宽估计函数 — 基于已观测到的带宽样本做调和平均。

    论文语义：在处理第 sample_idx 个样本时，只能利用之前已经观测到的
    带宽信息来估计当前带宽，而不是直接使用当前真实带宽点值。

    逻辑：
      - 若 sample_idx == 0，尚无历史观测，返回冷启动值 cold_start_bw
      - 若 sample_idx > 0，对 bandwidth_series[:sample_idx] 中循环映射到的
        前 sample_idx 个带宽值做调和平均：
          B = n / Σ(1/b_i)

    Args:
        bandwidth_series: list[float], 完整带宽序列（已做 bw_floor 保护）
        sample_idx:       int, 当前样本索引（0-based）
        cold_start_bw:    float, 冷启动带宽估计值（Mbps）
                          论文值：LTE = 7.6, 5G = 14.7

    Returns:
        float: 估计带宽 (Mbps)
    """
    if sample_idx == 0:
        return cold_start_bw

    # 收集 sample_idx 之前已观测到的带宽值（循环映射）
    n = sample_idx
    series_len = len(bandwidth_series)
    reciprocal_sum = 0.0
    for i in range(n):
        b_i = bandwidth_series[i % series_len]
        reciprocal_sum += 1.0 / b_i

    # 调和平均：B = n / Σ(1/b_i)
    estimated_bw = n / reciprocal_sum
    return estimated_bw


# =====================================================================
# 统一推理函数（用于准确率计算）
# =====================================================================

def run_pruned_vit_inference(model, image, token_schedule, N):
    """
    在本地执行 pruned ViT 完整推理（所有 N 层 + norm + head），
    按 token_schedule 进行固定 pruning。

    device_only / cloud_only / mixed 的推理逻辑完全相同，
    仅语义上区分 device/cloud，因此统一收敛到此函数。

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


# =====================================================================
# 汇总指标
# =====================================================================

def summarize(results, method, network_type, scenario):
    """
    计算汇总指标，包含场景元信息。

    Args:
        results:      list[dict], 推理结果
        method:       str, 方法名称（"device_only" / "cloud_only" / "mixed" / "janus"）
        network_type: str, "LTE" / "5G"
        scenario:     str, "static" / "walking" / "driving"

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
        "method": method,
        "num_samples": n,
        "avg_device_ms": avg_device_ms,
        "avg_cloud_ms": avg_cloud_ms,
        "avg_comm_ms": avg_comm_ms,
        "avg_total_ms": avg_total_ms,
        "accuracy": accuracy,
        "violation_ratio": violation_ratio,
        "throughput_fps": throughput_fps,
    }


# =====================================================================
# CSV 保存
# =====================================================================

def save_records_csv(results, method, network_type, scenario, output_path,
                     extra_columns=None):
    """
    将结果保存为带场景标签的逐样本明细 CSV。

    调用方须确保 results[i] 中已有以下两个带宽键：
      - "observed_bandwidth_mbps":  当前 trace 时间步的真实带宽点值
      - "estimated_bandwidth_mbps": 基于调和平均的带宽估计值（用于计算 comm_ms）
    所有四个脚本均通过 load_bandwidth_series() + get_bandwidth_for_sample() +
    estimate_bandwidth() 在写入前完成带宽字段填充。

    Args:
        results:        list[dict], 推理结果
        method:         str, 方法名称
        network_type:   str, "LTE" / "5G"
        scenario:       str, "static" / "walking" / "driving"
        output_path:    str, 输出文件路径
        extra_columns:  list[str] | None, 额外要写入的列名（如 ["alpha", "split_layer"]）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns = list(RECORD_COLUMNS)
    if extra_columns:
        columns.extend(extra_columns)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in results:
            row = [
                r["sample_id"],
                network_type,
                scenario,
                f"{r['observed_bandwidth_mbps']:.4f}",
                f"{r['estimated_bandwidth_mbps']:.4f}",
                method,
                f"{r['device_ms']:.4f}",
                f"{r['cloud_ms']:.4f}",
                f"{r['comm_ms']:.4f}",
                f"{r['total_time_ms']:.4f}",
                r["correct"],
                r["pred_label"],
                r["true_label"],
                r["violate"],
            ]
            if extra_columns:
                for col in extra_columns:
                    val = r[col]
                    # 浮点列格式化
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(val)
            writer.writerow(row)

    print(f"  [Save] records -> {output_path}")


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
