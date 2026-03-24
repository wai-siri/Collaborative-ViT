"""
带宽预测准确性验证脚本（滑动窗口调和平均）
=============================================
功能：
  使用滑动窗口调和平均预测下一时刻带宽，验证预测准确性。
  对每个网络场景，逐时间步计算：
    - observed_bw:  当前时间步的真实带宽
    - estimated_bw: 基于最近 W 个历史观测的调和平均估计带宽
  然后统计多种误差指标，并输出可视化图表。

预测逻辑：
  - idx == 0：采用单点冷启动值 cold_start_bw
  - idx > 0：对最近 W 个历史观测做调和平均
    B_hat = k / Σ(1/b_i)，其中 k = min(idx, W)
  - 窗口大小由 WINDOW_SIZE 控制（默认 30）

评估指标：
  - MAE:   Mean Absolute Error (Mbps)
  - RMSE:  Root Mean Squared Error (Mbps)
  - MAPE:  Mean Absolute Percentage Error (%)
  - MdAPE: Median Absolute Percentage Error (%)
  - R²:    决定系数（越接近 1 越好）
  - 带宽预测偏差分布（高估 / 低估比例）

输出（默认 results/bandwidth_validation/）：
  - bandwidth_validation_summary.csv            — 6 个场景的汇总误差指标
  - bandwidth_validation_{tag}_w{W}_detail.csv   — 逐时间步的预测明细
  - bandwidth_prediction_{tag}_w{W}.png          — 预测 vs 真实带宽对比图
  - bandwidth_error_summary_w{W}.png             — 各场景误差指标对比柱状图

用法：
  python src/utils/validate_bandwidth_prediction.py
"""

import os
import sys
import math
import csv
from collections import deque

# ── 路径设置 ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SRC_DIR)

import config
from simulation.baseline_common import (
    NETWORK_SCENARIOS,
    load_bandwidth_series,
    get_bandwidth_for_sample,
)

# ── 滑动窗口配置（与 config.BANDWIDTH_WINDOW_SIZE 统一） ──────────────
WINDOW_SIZE = config.BANDWIDTH_WINDOW_SIZE

# ── 可选依赖：matplotlib / numpy ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，适用于无 GUI 环境
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warning] matplotlib 未安装，跳过图表生成。安装方式: pip install matplotlib")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("[Warning] numpy 未安装，部分统计指标将使用纯 Python 计算。")


# =====================================================================
# 误差指标计算
# =====================================================================

def compute_metrics(observed_list, estimated_list):
    """
    计算预测准确性指标。

    Args:
        observed_list:  list[float], 真实带宽序列
        estimated_list: list[float], 预测带宽序列

    Returns:
        dict: 包含 MAE, RMSE, MAPE, MdAPE, R², overestimate_ratio, underestimate_ratio
    """
    n = len(observed_list)
    assert n == len(estimated_list) and n > 0

    abs_errors = []
    sq_errors = []
    ape_list = []       # Absolute Percentage Error
    over_count = 0      # 高估次数
    under_count = 0     # 低估次数

    for obs, est in zip(observed_list, estimated_list):
        err = est - obs
        abs_err = abs(err)
        abs_errors.append(abs_err)
        sq_errors.append(err ** 2)

        # 百分比误差（obs > 0 才有意义，trace 已做 floor 保护）
        if obs > 0:
            ape_list.append(abs_err / obs * 100.0)

        if err > 0:
            over_count += 1
        elif err < 0:
            under_count += 1

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    mape = sum(ape_list) / len(ape_list) if ape_list else 0.0

    # MdAPE: 中位数百分比误差
    sorted_ape = sorted(ape_list)
    if sorted_ape:
        mid = len(sorted_ape) // 2
        if len(sorted_ape) % 2 == 0:
            mdape = (sorted_ape[mid - 1] + sorted_ape[mid]) / 2.0
        else:
            mdape = sorted_ape[mid]
    else:
        mdape = 0.0

    # R² (coefficient of determination)
    obs_mean = sum(observed_list) / n
    ss_res = sum(sq_errors)
    ss_tot = sum((obs - obs_mean) ** 2 for obs in observed_list)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mdape": mdape,
        "r_squared": r_squared,
        "overestimate_ratio": over_count / n,
        "underestimate_ratio": under_count / n,
        "exact_ratio": (n - over_count - under_count) / n,
    }


# =====================================================================
# 可视化
# =====================================================================

def plot_prediction_vs_actual(observed, estimated, tag, output_path,
                              max_points=2000):
    """
    绘制预测带宽 vs 真实带宽的对比折线图。

    Args:
        observed:    list[float], 真实带宽
        estimated:   list[float], 预测带宽
        tag:         str, 场景标签（如 "lte_static"）
        output_path: str, 图片保存路径
        max_points:  int, 最大绘制点数（避免图过密）
    """
    if not HAS_MPL:
        return

    n = len(observed)
    # 降采样以保持图表清晰
    if n > max_points:
        step = n // max_points
        indices = list(range(0, n, step))
        obs_plot = [observed[i] for i in indices]
        est_plot = [estimated[i] for i in indices]
    else:
        indices = list(range(n))
        obs_plot = observed
        est_plot = estimated

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # 上子图：真实 vs 预测
    ax1 = axes[0]
    ax1.plot(indices, obs_plot, alpha=0.5, linewidth=0.8, label="Observed BW", color="#2196F3")
    ax1.plot(indices, est_plot, alpha=0.8, linewidth=1.0, label=f"Estimated BW (Sliding W={WINDOW_SIZE})", color="#FF5722")
    ax1.set_ylabel("Bandwidth (Mbps)")
    ax1.set_title(f"Bandwidth Prediction Validation — {tag}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 下子图：预测误差
    ax2 = axes[1]
    errors = [est_plot[i] - obs_plot[i] for i in range(len(obs_plot))]
    ax2.fill_between(indices, errors, 0, where=[e >= 0 for e in errors],
                     alpha=0.4, color="#4CAF50", label="Overestimate")
    ax2.fill_between(indices, errors, 0, where=[e < 0 for e in errors],
                     alpha=0.4, color="#F44336", label="Underestimate")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error (Mbps)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {output_path}")


def plot_error_summary(all_metrics, output_path):
    """
    绘制各场景误差指标对比柱状图（MAE, RMSE, MAPE）。

    Args:
        all_metrics: list[dict], 每个元素包含 tag + compute_metrics 的返回值
        output_path: str, 图片保存路径
    """
    if not HAS_MPL:
        return

    tags = [m["tag"] for m in all_metrics]
    mae_vals = [m["mae"] for m in all_metrics]
    rmse_vals = [m["rmse"] for m in all_metrics]
    mape_vals = [m["mape"] for m in all_metrics]
    r2_vals = [m["r_squared"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = range(len(tags))
    bar_width = 0.5

    # MAE
    ax = axes[0][0]
    bars = ax.bar(x, mae_vals, bar_width, color="#2196F3", alpha=0.8)
    ax.set_ylabel("MAE (Mbps)")
    ax.set_title("Mean Absolute Error")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # RMSE
    ax = axes[0][1]
    bars = ax.bar(x, rmse_vals, bar_width, color="#FF9800", alpha=0.8)
    ax.set_ylabel("RMSE (Mbps)")
    ax.set_title("Root Mean Squared Error")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # MAPE
    ax = axes[1][0]
    bars = ax.bar(x, mape_vals, bar_width, color="#4CAF50", alpha=0.8)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Mean Absolute Percentage Error")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars, mape_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # R²
    ax = axes[1][1]
    colors = ["#4CAF50" if v > 0.5 else "#FF5722" for v in r2_vals]
    bars = ax.bar(x, r2_vals, bar_width, color=colors, alpha=0.8)
    ax.set_ylabel("R²")
    ax.set_title("Coefficient of Determination")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Sliding Window (W={WINDOW_SIZE}) Bandwidth Prediction — Error Summary", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] {output_path}")


# =====================================================================
# CSV 保存
# =====================================================================

DETAIL_COLUMNS = [
    "sample_idx", "observed_bw_mbps", "estimated_bw_mbps",
    "error_mbps", "abs_error_mbps", "ape_percent",
]

SUMMARY_COLUMNS = [
    "network_type", "scenario", "tag", "window_size", "num_samples",
    "mae_mbps", "rmse_mbps", "mape_percent", "mdape_percent",
    "r_squared", "overestimate_ratio", "underestimate_ratio", "exact_ratio",
]


def save_detail_csv(records, output_path):
    """保存逐时间步预测明细 CSV。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(DETAIL_COLUMNS)
        for r in records:
            writer.writerow([
                r["sample_idx"],
                f"{r['observed_bw_mbps']:.4f}",
                f"{r['estimated_bw_mbps']:.4f}",
                f"{r['error_mbps']:.4f}",
                f"{r['abs_error_mbps']:.4f}",
                f"{r['ape_percent']:.4f}",
            ])
    print(f"  [Save] detail -> {output_path}")


def save_summary_csv(all_metrics, output_path):
    """保存各场景汇总误差指标 CSV。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for m in all_metrics:
            writer.writerow([
                m["network_type"],
                m["scenario"],
                m["tag"],
                m["window_size"],
                m["n"],
                f"{m['mae']:.4f}",
                f"{m['rmse']:.4f}",
                f"{m['mape']:.4f}",
                f"{m['mdape']:.4f}",
                f"{m['r_squared']:.6f}",
                f"{m['overestimate_ratio']:.6f}",
                f"{m['underestimate_ratio']:.6f}",
                f"{m['exact_ratio']:.6f}",
            ])
    print(f"  [Save] summary -> {output_path}")


# =====================================================================
# 主流程
# =====================================================================

def main():
    trace_dir = os.path.join(config.PROJECT_ROOT, config.NETWORK_TRACE_DIR)
    output_dir = os.path.join(config.PROJECT_ROOT, "results", "bandwidth_validation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Bandwidth Prediction Validation (Sliding Window, W={WINDOW_SIZE})")
    print("=" * 60)

    all_metrics = []

    for sc in NETWORK_SCENARIOS:
        net_type = sc["network_type"]
        scenario = sc["scenario"]
        tag = f"{net_type.lower()}_{scenario}_w{WINDOW_SIZE}"
        cold_start_bw = sc["cold_start_bw"]
        bw_floor = sc["bw_floor"]

        # 1. 读取带宽 trace
        trace_path = os.path.join(trace_dir, sc["trace_file"])
        if not os.path.exists(trace_path):
            print(f"  [Skip] {trace_path} 不存在，跳过")
            continue

        bw_series = load_bandwidth_series(trace_path, bw_floor=bw_floor)
        n = len(bw_series)

        print(f"\n[{tag}] trace_len={n}, cold_start_bw={cold_start_bw} Mbps, "
              f"bw_floor={bw_floor} Mbps, window_size={WINDOW_SIZE}")

        # 2. 逐时间步计算预测带宽 vs 真实带宽
        #    使用滑动窗口调和平均预测下一时刻带宽
        #    窗口大小由 WINDOW_SIZE 控制，idx=0 采用单点冷启动值
        observed_list = []
        estimated_list = []
        detail_records = []
        history_window = deque()  # 最近 W 个历史带宽观测值

        for idx in range(n):
            obs_bw = get_bandwidth_for_sample(bw_series, idx)

            # 滑动窗口调和平均预测
            if idx == 0:
                est_bw = cold_start_bw
            else:
                # 对 history_window 中的值做调和平均：B = k / Σ(1/b_i)
                k = len(history_window)
                reciprocal_sum = sum(1.0 / b for b in history_window)
                est_bw = k / reciprocal_sum

            # 将当前观测放入窗口（供下一步使用）
            history_window.append(obs_bw)
            if len(history_window) > WINDOW_SIZE:
                history_window.popleft()

            error = est_bw - obs_bw
            abs_error = abs(error)
            ape = (abs_error / obs_bw * 100.0) if obs_bw > 0 else 0.0

            observed_list.append(obs_bw)
            estimated_list.append(est_bw)
            detail_records.append({
                "sample_idx": idx,
                "observed_bw_mbps": obs_bw,
                "estimated_bw_mbps": est_bw,
                "error_mbps": error,
                "abs_error_mbps": abs_error,
                "ape_percent": ape,
            })

        # 3. 计算误差指标（跳过 idx=0 冷启动样本，从 idx=1 开始评估）
        metrics = compute_metrics(observed_list[1:], estimated_list[1:])
        metrics["network_type"] = net_type
        metrics["scenario"] = scenario
        metrics["tag"] = tag
        metrics["window_size"] = WINDOW_SIZE
        all_metrics.append(metrics)

        # 4. 控制台输出
        print(f"  MAE  = {metrics['mae']:.4f} Mbps")
        print(f"  RMSE = {metrics['rmse']:.4f} Mbps")
        print(f"  MAPE = {metrics['mape']:.2f}%")
        print(f"  MdAPE= {metrics['mdape']:.2f}%")
        print(f"  R²   = {metrics['r_squared']:.6f}")
        print(f"  高估比例 = {metrics['overestimate_ratio']:.2%}, "
              f"低估比例 = {metrics['underestimate_ratio']:.2%}")

        # 5. 保存逐时间步明细 CSV
        detail_path = os.path.join(output_dir, f"bandwidth_validation_{tag}_detail.csv")
        save_detail_csv(detail_records, detail_path)

        # 6. 绘制预测 vs 真实对比图
        plot_path = os.path.join(output_dir, f"bandwidth_prediction_{tag}.png")
        plot_prediction_vs_actual(observed_list, estimated_list, tag, plot_path)

    # ── 保存汇总 CSV ──
    if all_metrics:
        summary_path = os.path.join(output_dir, "bandwidth_validation_summary.csv")
        save_summary_csv(all_metrics, summary_path)

        # ── 绘制误差汇总对比图 ──
        error_plot_path = os.path.join(output_dir, f"bandwidth_error_summary_w{WINDOW_SIZE}.png")
        plot_error_summary(all_metrics, error_plot_path)

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)
    if all_metrics:
        avg_mape = sum(m["mape"] for m in all_metrics) / len(all_metrics)
        avg_r2 = sum(m["r_squared"] for m in all_metrics) / len(all_metrics)
        print(f"  全场景平均 MAPE = {avg_mape:.2f}%")
        print(f"  全场景平均 R²   = {avg_r2:.6f}")

        # 给出定性结论
        if avg_mape < 10:
            print("  结论：滑动窗口调和平均预测整体表现优秀（MAPE < 10%）")
        elif avg_mape < 25:
            print("  结论：滑动窗口调和平均预测整体表现良好（10% < MAPE < 25%）")
        elif avg_mape < 50:
            print("  结论：滑动窗口调和平均预测整体表现一般（25% < MAPE < 50%）")
        else:
            print("  结论：滑动窗口调和平均预测整体偏差较大（MAPE > 50%），建议考虑改进方法")

        # 分析高估/低估偏差倾向
        avg_over = sum(m["overestimate_ratio"] for m in all_metrics) / len(all_metrics)
        avg_under = sum(m["underestimate_ratio"] for m in all_metrics) / len(all_metrics)
        if avg_over > avg_under * 1.2:
            print(f"  偏差倾向：整体偏向高估（高估 {avg_over:.1%} vs 低估 {avg_under:.1%}）")
        elif avg_under > avg_over * 1.2:
            print(f"  偏差倾向：整体偏向低估（低估 {avg_under:.1%} vs 高估 {avg_over:.1%}）")
        else:
            print(f"  偏差倾向：高估与低估大致均衡（高估 {avg_over:.1%} vs 低估 {avg_under:.1%}）")

    print(f"\n  输出目录: {output_dir}")
    file_count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
    print(f"  共生成 {file_count} 个文件")


if __name__ == "__main__":
    main()
