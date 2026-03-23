"""
Device-Only Baseline — Fig.7 结果生成器
=======================================
功能：
  - 固定每层裁剪 23 tokens 的 baseline pruning（论文 Fig.7 口径）
  - 语义上等价于 Janus split_layer = N+1（全部在 device 端执行，cloud_ms = 0, comm_ms = 0）
  - 使用 profiler 线性模型预测 device 端逐层延迟（不使用真实计时）
  - 使用真实 forward 得到分类结果计算准确率
  - 只执行一次推理，然后展开到 6 个网络场景（LTE/5G × static/walking/driving）
    方便与 Cloud-Only / Mixed / Janus 统一对比

输出目录（默认 results/device_only/）：
  每个场景生成两个文件：
    device_only_{scenario}_records.csv   — 逐样本明细
    device_only_{scenario}_summary.csv   — 汇总指标

用法：
  python src/simulation/device_only.py
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
    predict_device_time, run_pruned_vit_inference,
    load_bandwidth_series, get_bandwidth_for_sample, estimate_bandwidth,
    summarize, save_records_csv, save_summary_csv,
)

METHOD = "device_only"


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    data_path = os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH)
    sla = config.DEFAULT_SLA
    prune_per_layer = config.PRUNE_PER_LAYER
    output_dir = os.path.join(config.PROJECT_ROOT, "results", "device_only")

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
    token_schedule = build_fixed_baseline_schedule(N, x_0, prune_per_layer)
    print(f"[Model]  {config.MODEL_NAME}, N={N}, x_0={x_0}, device={dev}")
    print(f"[Config] prune_per_layer={prune_per_layer}, SLA={sla}ms")
    print(f"[Schedule] layer 1 keep={token_schedule[1]}, "
          f"layer {N} keep={token_schedule[N]}")

    # 1-3. 加载数据集
    loader = get_imagenet_loader(data_path, batch_size=1, num_workers=0)
    total_samples = len(loader.dataset)
    print(f"[Data]   {total_samples} samples loaded")

    # 1-4. 逐样本推理
    device_ms_value = predict_device_time(token_schedule, N)
    base_results = []
    for sample_idx, (images, labels) in enumerate(
            tqdm(loader, desc="Device-Only Inference", total=total_samples)): # 进度条包装器
        images = images.to(dev)
        label = int(labels.item())

        with torch.no_grad():
            pred_label, _ = run_pruned_vit_inference(
                model, images, token_schedule, N)

        correct = 1 if pred_label == label else 0
        violate = 1 if device_ms_value > sla else 0
        base_results.append({
            "sample_id": sample_idx,
            "device_ms": device_ms_value,
            "cloud_ms": 0.0,
            "comm_ms": 0.0,
            "total_time_ms": device_ms_value,
            "correct": correct,
            "pred_label": pred_label,
            "true_label": label,
            "violate": violate,
        })

    # ── 控制台输出概览 ──
    tmp_summary = summarize(base_results, METHOD, "-", "-")
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
    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)

    print(f"\n[Phase 2] 展开到 {len(NETWORK_SCENARIOS)} 个网络场景 ...")
    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}"

        # 读取该场景完整带宽序列
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        bw_series = load_bandwidth_series(trace_path, bw_floor=sc["bw_floor"])

        cold_start_bw = sc["cold_start_bw"]

        # 逐样本复制 base 记录，并按样本索引映射带宽
        scenario_results = []
        for r in base_results:
            rec = dict(r)  # 浅拷贝一份
            sid = rec["sample_id"]
            rec["observed_bandwidth_mbps"] = get_bandwidth_for_sample(
                bw_series, sid)
            rec["estimated_bandwidth_mbps"] = estimate_bandwidth(
                bw_series, sid, cold_start_bw)
            scenario_results.append(rec)

        # 保存逐样本明细
        records_path = os.path.join(output_dir, f"device_only_{tag}_records.csv")
        save_records_csv(scenario_results, METHOD, net_type, scenario, records_path)

        # 计算并保存汇总
        summary = summarize(scenario_results, METHOD, net_type, scenario)
        summary_path = os.path.join(output_dir, f"device_only_{tag}_summary.csv")
        save_summary_csv(summary, summary_path)

    print(f"\n[Done] 共保存 {len(NETWORK_SCENARIOS) * 2} 个文件到 {output_dir}")


if __name__ == "__main__":
    main()