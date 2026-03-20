"""
Mixed Baseline — Fig.7 结果生成器
=================================
功能：
  - 对每个样本 / 每个网络条件，在 Device-Only 和 Cloud-Only 之间二选一
  - 选择预测时延更小的路径，整条样本走该路径
  - 不是 split computing，不是动态 split point，是 binary selector
  - 固定每层裁剪 23 tokens 的 baseline pruning（论文 Fig.7 口径）
  - 使用 profiler 线性模型分别预测 device / cloud 端逐层延迟
  - comm_ms = 初始 token 张量大小 (x_0 * D_M * bits) / 动态带宽（仅 cloud 路径时 > 0）
    与 Janus split_layer=0 完全一致，传输的是 embedding 后的 token tensor
  - 使用真实 forward 得到分类结果计算准确率
  - 逐网络场景、逐样本独立判断路径并执行推理

输出目录（默认 results/mixed/）：
  每个场景生成两个文件：
    mixed_{scenario}_records.csv   — 逐样本明细
    mixed_{scenario}_summary.csv   — 汇总指标

路径选择规则：
  - 选 device-only 时：device_ms > 0, cloud_ms = 0, comm_ms = 0
  - 选 cloud-only 时：device_ms = 0, cloud_ms > 0, comm_ms > 0

用法：
  python src/simulation/mixed.py [--data_path DATA] [--sla SLA_MS]
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
from schedule.schedule import init, device_profiler, cloud_profiler
from schedule.token_pruning import prune_tokens
from schedule.split_inference import _embed
from utils.imagenet_loader import get_imagenet_loader


# =====================================================================
# 第二部分：常量与通用函数
# =====================================================================

# ── 6 个网络场景（与 Device-Only / Cloud-Only / Janus 保持一致） ──────
NETWORK_SCENARIOS = [
    {"network_type": "LTE", "scenario": "static",  "trace_file": "static_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "walking", "trace_file": "walking_lte_trace.csv"},
    {"network_type": "LTE", "scenario": "driving", "trace_file": "driving_lte_trace.csv"},
    {"network_type": "5G",  "scenario": "static",  "trace_file": "static_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "walking", "trace_file": "walking_5g_trace.csv"},
    {"network_type": "5G",  "scenario": "driving", "trace_file": "driving_5g_trace.csv"},
]

METHOD = "mixed"


# ── 2.2 固定每层裁剪 baseline 的 token schedule ──────────────────────
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


# ── 2.3 预测 device-only 推理时间（device profiler 线性模型） ─────────
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


# ── 2.4 预测 cloud-only 推理时间（cloud profiler 线性模型） ──────────
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


# ── 2.5 cloud 通信相关 ───────────────────────────────────────────────
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


def predict_tensor_transfer_time(x_0, D_M, bits, bandwidth_mbps):
    """
    计算 cloud 路径的通信时间：传输 embedding 后的 token tensor。
    comm_ms = x_0 * D_M * bits / (bandwidth_mbps * 1e6) * 1000

    与 Janus scheduler 中 split_layer=0 的通信量计算方式完全一致，
    确保 Mixed baseline 的 cloud 路径与 Janus cloud-only 退化路径口径统一。

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


# ── 2.6 两条真实推理函数 ─────────────────────────────────────────────
def run_device_only_inference(model, image, token_schedule, N):
    """
    在 device 端执行完整推理（所有 N 层 + norm + head），
    按 token_schedule 进行固定 pruning。

    注意：当前实现与 run_cloud_only_inference 完全相同，
    两者仅在语义上区分 device/cloud，执行逻辑一致。
    保留两个函数是为了与 device_only.py / cloud_only.py 对齐，
    便于未来各自独立演化。

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


def run_cloud_only_inference(model, image, token_schedule, N):
    """
    逻辑上表示云端完整执行所有 N 层 + norm + head，
    按 token_schedule 进行固定 pruning。
    实际在本地执行 forward，仅用于得到准确率。

    注意：当前实现与 run_device_only_inference 完全相同，
    两者仅在语义上区分 device/cloud，执行逻辑一致。
    保留两个函数是为了与 device_only.py / cloud_only.py 对齐，
    便于未来各自独立演化。

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


# ── 2.7 路径选择函数 ─────────────────────────────────────────────────
def choose_path(device_total_ms, cloud_total_ms):
    """
    在 Device-Only 和 Cloud-Only 之间选择预测时延更小的路径。

    Args:
        device_total_ms: float, device-only 预测总时延（毫秒）
        cloud_total_ms:  float, cloud-only 预测总时延（毫秒）

    Returns:
        str: "device_only" 或 "cloud_only"
    """
    if device_total_ms <= cloud_total_ms:
        return "device_only"
    else:
        return "cloud_only"


# =====================================================================
# 第三部分：单样本 mixed 评估函数
# =====================================================================

def evaluate_one_sample_mixed(model, image, label, sample_id,
                              token_schedule, N, x_0, D_M, bits,
                              bandwidth_mbps, sla_ms,
                              device_pred_ms, cloud_pred_ms):
    """
    对单张图片完成 Mixed 二选一评估：
      1. 算 device 候选总时延
      2. 算 cloud 候选总时延（含通信）
      3. 比较两者，选更小者
      4. 按被选路径做真实推理
      5. 生成统一格式记录

    Args:
        model:          timm ViT 模型
        image:          Tensor (1, 3, 384, 384)
        label:          int, 真实标签
        sample_id:      int, 样本索引
        token_schedule: dict, {l: keep_n}
        N:              int, Transformer 层数
        x_0:            int, 初始 token 数（含 CLS）
        D_M:            int, token embedding 维度
        bits:           int, 数据类型占用 bit 数
        bandwidth_mbps: float, 当前样本对应的带宽 (Mbps)
        sla_ms:         float, SLA 延迟上限（毫秒）
        device_pred_ms: float, 预计算的 device-only 推理时延（毫秒），整次运行恒定
        cloud_pred_ms:  float, 预计算的 cloud-only 推理时延（毫秒），整次运行恒定

    Returns:
        dict: 统一格式记录，字段与 device_only / cloud_only 完全一致
    """
    # ── 第 1 步：算 device 候选总时延 ──
    device_ms = device_pred_ms
    device_total_ms = device_ms

    # ── 第 2 步：算 cloud 候选总时延（通信量按 token tensor 计算） ──
    cloud_ms = cloud_pred_ms
    comm_ms = predict_tensor_transfer_time(x_0, D_M, bits, bandwidth_mbps)
    cloud_total_ms = cloud_ms + comm_ms

    # ── 第 3 步：比较两者 ──
    path = choose_path(device_total_ms, cloud_total_ms)

    # ── 第 4 步：按被选路径做真实推理 ──
    with torch.no_grad():
        if path == "device_only":
            pred_label, _ = run_device_only_inference(model, image, token_schedule, N)
            final_device_ms = device_ms
            final_cloud_ms = 0.0
            final_comm_ms = 0.0
            final_total_ms = device_total_ms
        else:
            pred_label, _ = run_cloud_only_inference(model, image, token_schedule, N)
            final_device_ms = 0.0
            final_cloud_ms = cloud_ms
            final_comm_ms = comm_ms
            final_total_ms = cloud_total_ms

    # ── 第 5 步：生成统一格式记录 ──
    correct = 1 if pred_label == label else 0
    violate = 1 if final_total_ms > sla_ms else 0

    return {
        "sample_id": sample_id,
        "device_ms": final_device_ms,
        "cloud_ms": final_cloud_ms,
        "comm_ms": final_comm_ms,
        "total_time_ms": final_total_ms,
        "correct": correct,
        "pred_label": pred_label,
        "true_label": label,
        "violate": violate,
        "bandwidth_mbps": bandwidth_mbps,
    }


# =====================================================================
# 第四部分：汇总与保存
# =====================================================================

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


# =====================================================================
# 第五部分：主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mixed Baseline — Fig.7 Result Generator (binary selector)")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH),
                        help="ImageNet parquet 文件路径")
    parser.add_argument("--sla", type=float, default=config.DEFAULT_SLA,
                        help="SLA 延迟上限（毫秒），默认 300ms")
    parser.add_argument("--prune_per_layer", type=int, default=23,
                        help="每层固定裁剪 token 数（论文 Fig.7 baseline = 23）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认 results/mixed/）")
    args = parser.parse_args()

    # ================================================================
    # Phase 1: 初始化
    # ================================================================
    print("=" * 60)
    print("Mixed Baseline — Fig.7 Result Generator (binary selector)")
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

    # 1-3. 预览 device / cloud profiler 预测时间
    device_pred_ms = predict_device_time(token_schedule, N)
    cloud_pred_ms = predict_cloud_time(token_schedule, N)
    print(f"[Profiler] device_pred_ms = {device_pred_ms:.4f} ms")
    print(f"[Profiler] cloud_pred_ms  = {cloud_pred_ms:.4f} ms")
    print(f"[Logic]  对每个样本: 若 device_total <= cloud_total → 选 device, 否则选 cloud")

    print(f"[Comm]   cloud 路径通信大小按 token tensor 计算: x_0={x_0} * D_M={D_M} * bits={bits} = {x_0*D_M*bits} bits")

    # 1-4. 加载数据集
    loader = get_imagenet_loader(args.data_path, batch_size=1, num_workers=0)
    dataset = loader.dataset
    total_samples = len(dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # ================================================================
    # Phase 2: 逐网络场景直接跑
    # ================================================================
    if args.output_dir is None:
        output_dir = os.path.join(config.PROJECT_ROOT, "results", "mixed")
    else:
        output_dir = args.output_dir

    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 逐网络场景遍历数据集，每样本二选一 ...")
    print(f"  共 {len(NETWORK_SCENARIOS)} 个场景 × {total_samples} 个样本")

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
        device_count = 0
        cloud_count = 0

        for sample_idx, (images, labels) in enumerate(
                tqdm(loader, desc=f"Mixed [{tag}]", total=total_samples)):
            images = images.to(dev)
            label = int(labels.item())

            # 2-3. 当前样本取当前带宽
            bandwidth_mbps = get_bandwidth_for_sample(bw_series, sample_idx)

            # 2-4. 调用单样本 mixed 评估
            record = evaluate_one_sample_mixed(
                model, images, label, sample_idx,
                token_schedule, N, x_0, D_M, bits,
                bandwidth_mbps, args.sla,
                device_pred_ms, cloud_pred_ms)

            results.append(record)

            # 统计路径选择（仅用于控制台输出）
            if record["cloud_ms"] == 0.0 and record["comm_ms"] == 0.0:
                device_count += 1
            else:
                cloud_count += 1

        # 2-5. 场景跑完：保存 records
        records_path = os.path.join(output_dir, f"mixed_{tag}_records.csv")
        save_records_csv(results, net_type, scenario, records_path)

        # 2-6. 计算并保存 summary
        summary = summarize(results, net_type, scenario)
        summary_path = os.path.join(output_dir, f"mixed_{tag}_summary.csv")
        save_summary_csv(summary, summary_path)

        # 控制台输出该场景概览
        print(f"    Path Selection : device={device_count}, cloud={cloud_count}")
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
