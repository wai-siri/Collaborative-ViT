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
  python src/simulation/cloud_only.py
  所有参数通过 config.py 统一配置
"""

import os
import sys

# ── 路径设置 ──────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

import torch
from tqdm import tqdm

import config
from schedule.schedule import init
from utils.imagenet_loader import get_imagenet_loader
from simulation.baseline_common import (
    NETWORK_SCENARIOS, build_fixed_baseline_schedule,
    predict_cloud_time, predict_tensor_transfer_time,
    load_bandwidth_series, get_bandwidth_for_sample, estimate_bandwidth,
    run_pruned_vit_inference, summarize, save_records_csv, save_summary_csv,
)

METHOD = "cloud_only"


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    data_path = os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH)
    sla = config.DEFAULT_SLA
    prune_per_layer = config.PRUNE_PER_LAYER
    output_dir = os.path.join(config.PROJECT_ROOT, "results", "cloud_only")

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
    token_schedule = build_fixed_baseline_schedule(N, x_0, prune_per_layer)
    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, device={dev}")
    print(f"[Config] prune_per_layer={prune_per_layer}, SLA={sla}ms")
    print(f"[Schedule] layer 1 keep={token_schedule[1]}, "
          f"layer {N} keep={token_schedule[N]}")
    print(f"[Comm]   通信大小按 token tensor 计算: x_0={x_0} * D_M={D_M} * bits={bits} = {x_0*D_M*bits} bits")

    # 1-3. 加载数据集
    loader = get_imagenet_loader(data_path, batch_size=1, num_workers=0)
    total_samples = len(loader.dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # 1-4. 逐样本推理
    # cloud_ms 固定，comm_ms 由 token tensor 大小和动态带宽共同决定，留到 Phase 2
    cloud_ms_value = predict_cloud_time(token_schedule, N)
    print(f"[Profiler] cloud_ms (all samples) = {cloud_ms_value:.4f} ms")

    inference_cache = []
    for sample_idx, (images, labels) in enumerate(
            tqdm(loader, desc="Cloud-Only Inference", total=total_samples)):
        images = images.to(dev)
        label = int(labels.item())

        with torch.no_grad():
            pred_label, _ = run_pruned_vit_inference(
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
    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 按 {len(NETWORK_SCENARIOS)} 个网络场景分别计算时延 ...")
    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}"
        cold_start_bw = sc["cold_start_bw"]

        # 读取该场景完整带宽序列
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bw_series = load_bandwidth_series(trace_path, bw_floor=sc["bw_floor"])
        avg_bw = sum(bw_series) / len(bw_series)
        min_bw = min(bw_series)
        max_bw = max(bw_series)

        print(f"\n  [{tag}] trace_len={len(bw_series)}, "
              f"avg_bw={avg_bw:.2f}, min_bw={min_bw:.2f}, max_bw={max_bw:.2f} Mbps, "
              f"cold_start_bw={cold_start_bw} Mbps")

        # 为该场景逐样本计算 comm_ms（基于调和平均估计带宽 + token tensor 大小）
        scenario_results = []
        for cached in inference_cache:
            sid = cached["sample_id"]
            # 当前 trace 时间步的真实带宽点值（仅记录，不用于 comm_ms 计算）
            observed_bw = get_bandwidth_for_sample(bw_series, sid)
            # 基于历史观测的调和平均估计带宽（用于 comm_ms 计算）
            estimated_bw = estimate_bandwidth(bw_series, sid, cold_start_bw)
            comm_ms = predict_tensor_transfer_time(x_0, D_M, bits, estimated_bw)
            total_time_ms = cached["cloud_ms"] + comm_ms
            violate = 1 if total_time_ms > sla else 0

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
                "observed_bandwidth_mbps": observed_bw,
                "estimated_bandwidth_mbps": estimated_bw,
            })

        # 保存逐样本明细
        records_path = os.path.join(output_dir, f"cloud_only_{tag}_records.csv")
        save_records_csv(scenario_results, METHOD, net_type, scenario, records_path)

        # 计算并保存汇总
        summary = summarize(scenario_results, METHOD, net_type, scenario)
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
