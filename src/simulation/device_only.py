"""
Device-Only Baseline — Fig.7 结果生成器
=======================================
功能：
  - 固定每层裁剪 23 tokens 的 baseline pruning（论文 Fig.7 口径）
  - split_layer = N（全部在 device 端执行，cloud_ms = 0, comm_ms = 0）
  - 使用 profiler 线性模型预测 device 端逐层延迟（不使用真实计时）
  - 使用真实 forward 得到分类结果计算准确率
  - 只执行一次推理，然后展开到 6 个网络场景（LTE/5G × static/walking/driving）
    方便与 Cloud-Only / Mixed / Janus 统一对比

输出目录（默认 results/device_only/）：
  每个场景生成两个文件：
    device_only_{scenario}_records.csv   — 逐样本明细
    device_only_{scenario}_summary.csv   — 汇总指标

用法：
  python src/simulation/device_only.py [--data_path DATA] [--sla SLA_MS]
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
from schedule.schedule import init, device_profiler
from schedule.token_pruning import prune_tokens
from schedule.split_inference import _embed
from utils.imagenet_loader import get_imagenet_loader


# ── 6 个网络场景（结构化，与 Cloud-Only / Mixed / Janus 保持一致） ────
NETWORK_SCENARIOS = [
    {"network_type": "LTE", "scenario": "static",  "trace_file": "static_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "walking", "trace_file": "walking_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "driving", "trace_file": "driving_lte_trace.csv"},
    {"network_type": "5G",  "scenario": "static",  "trace_file": "static_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "walking", "trace_file": "walking_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "driving", "trace_file": "driving_5g_trace.csv"},
]

METHOD = "device_only"


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


# ── 预测 device-only 推理时间（profiler 线性模型） ──────────────────────
def predict_device_time(token_schedule, N):
    """
    使用 profiler 线性模型预测 device-only 整体推理时间。
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


# ── 实际推理（用于准确率计算） ──────────────────────────────────────────
def run_device_only_inference(model, image, token_schedule, N):
    """
    在 device 端执行完整推理（所有 N 层 + norm + head），
    按 token_schedule 进行固定 pruning。

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


# ── 单样本评估（统一字段格式） ─────────────────────────────────────────
def evaluate_one_sample(model, image, label, sample_id, token_schedule, N, sla_ms):
    """
    对单张图片执行：
      1. profiler 预测 device 端时间
      2. 真实推理得到预测标签
      3. 判断是否正确 & 是否违规

    Returns:
        dict: 统一格式，包含 device_ms, cloud_ms, comm_ms, total_time_ms,
              correct, pred_label, true_label, violate, sample_id
    """
    device_ms = predict_device_time(token_schedule, N)
    cloud_ms = 0.0   # Device-Only 不涉及 cloud 计算
    comm_ms = 0.0     # Device-Only 不涉及网络传输
    total_time_ms = device_ms + cloud_ms + comm_ms

    with torch.no_grad():
        pred_label, _ = run_device_only_inference(model, image, token_schedule, N)

    correct = 1 if pred_label == label else 0
    violate = 1 if total_time_ms > sla_ms else 0

    return {
        "sample_id": sample_id,
        "device_ms": device_ms,
        "cloud_ms": cloud_ms,
        "comm_ms": comm_ms,
        "total_time_ms": total_time_ms,
        "correct": correct,
        "pred_label": pred_label,
        "true_label": label,
        "violate": violate,
    }


# ── 读取网络 trace 的平均带宽 ─────────────────────────────────────────
def load_avg_bandwidth(trace_file):
    """
    从网络 trace CSV 中读取平均带宽（Mbps）。

    Args:
        trace_file: str, trace 文件的完整路径

    Returns:
        float: 平均带宽 (Mbps)
    """
    df = pd.read_csv(trace_file)
    return float(df["bandwidth_mbps"].mean())


# ── 汇总指标 ──────────────────────────────────────────────────────────
def summarize(results, sla_ms, network_type, scenario):
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

def save_records_csv(base_results, network_type, scenario, bandwidth_mbps, output_path):
    """
    将 base_results 扩展为带场景标签的逐样本明细并保存。

    Args:
        base_results:   list[dict], 推理结果（不含场景信息）
        network_type:   str, "LTE" / "5G"
        scenario:       str, "static" / "walking" / "driving"
        bandwidth_mbps: float, 该场景的平均带宽
        output_path:    str, 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        for r in base_results:
            writer.writerow([
                r["sample_id"],
                network_type,
                scenario,
                f"{bandwidth_mbps:.4f}",
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
        description="Device-Only Baseline — Fig.7 Result Generator")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH),
                        help="ImageNet parquet 文件路径")
    parser.add_argument("--sla", type=float, default=config.DEFAULT_SLA,
                        help="SLA 延迟上限（毫秒），默认 300ms")
    parser.add_argument("--prune_per_layer", type=int, default=23,
                        help="每层固定裁剪 token 数（论文 Fig.7 baseline = 23）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/device_only/）")
    args = parser.parse_args()

    # ================================================================
    # Phase 1: 执行一次 Device-Only 推理，得到 base_results
    # ================================================================
    print("=" * 60)
    print("Device-Only Baseline — Fig.7 Result Generator")
    print("=" * 60)

    # 1-1. 加载模型
    model = init()
    dev = next(model.parameters()).device
    N = len(model.blocks)           # 24
    x_0 = model.pos_embed.size(1)   # 577

    # 1-2. 构建固定 baseline pruning 计划
    token_schedule = build_fixed_baseline_schedule(N, x_0, args.prune_per_layer)
    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, device={dev}")
    print(f"[Config] prune_per_layer={args.prune_per_layer}, SLA={args.sla}ms")
    print(f"[Schedule] layer 1 keep={token_schedule[1]}, "
          f"layer {N} keep={token_schedule[N]}")

    # 1-3. 加载数据集
    loader = get_imagenet_loader(args.data_path, batch_size=1, num_workers=0)
    total_samples = len(loader.dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # 1-4. 逐样本推理（只跑一次）
    base_results = []
    for sample_idx, (images, labels) in enumerate(
            tqdm(loader, desc="Device-Only Inference", total=total_samples)):
        images = images.to(dev)
        label = int(labels.item())
        r = evaluate_one_sample(
            model, images, label, sample_idx, token_schedule, N, args.sla)
        base_results.append(r)

    # ── 控制台输出概览 ──
    tmp_summary = summarize(base_results, args.sla, "-", "-")
    print("\n===== Device-Only Results =====")
    print(f"  Avg Device Time (ms) : {tmp_summary['avg_device_ms']:.4f}")
    print(f"  Avg Cloud Time (ms)  : {tmp_summary['avg_cloud_ms']:.4f}")
    print(f"  Avg Comm Time (ms)   : {tmp_summary['avg_comm_ms']:.4f}")
    print(f"  Avg Total Time (ms)  : {tmp_summary['avg_total_ms']:.4f}")
    print(f"  Accuracy             : {tmp_summary['accuracy']:.6f}")
    print(f"  Violation Ratio      : {tmp_summary['violation_ratio']:.6f}")
    print(f"  Avg Throughput (fps) : {tmp_summary['throughput_fps']:.4f}")
    print("=" * 60)

    # ================================================================
    # Phase 2: 展开到 6 个网络场景，分别保存 records + summary
    # ================================================================
    if args.output_dir is None:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "device_only")
    else:
        output_dir = args.output_dir

    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 展开到 {len(NETWORK_SCENARIOS)} 个网络场景 ...")
    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}"

        # 读取该场景的平均带宽
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bandwidth_mbps = load_avg_bandwidth(trace_path)

        # 保存逐样本明细
        records_path = os.path.join(output_dir, f"device_only_{tag}_records.csv")
        save_records_csv(base_results, net_type, scenario, bandwidth_mbps, records_path)

        # 计算并保存汇总
        summary = summarize(base_results, args.sla, net_type, scenario)
        summary_path = os.path.join(output_dir, f"device_only_{tag}_summary.csv")
        save_summary_csv(summary, summary_path)

    print(f"\n[Done] 共保存 {len(NETWORK_SCENARIOS) * 2} 个文件到 {output_dir}")


if __name__ == "__main__":
    main()