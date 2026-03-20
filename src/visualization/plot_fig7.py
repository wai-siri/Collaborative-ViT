"""
plot_fig7.py  —  生成论文 Fig.7 风格的对比柱状图
==================================================
读取 results/ 下四种方法（device_only, cloud_only, mixed, janus）的
summary CSV，按网络类型（5G / LTE）分别绘制两幅图，每幅图包含：
  左子图: Violation Ratio (%)
  右子图: Avg. Throughput (FPS)

输出:
  results/fig7_5g.pdf   / fig7_5g.png
  results/fig7_lte.pdf  / fig7_lte.png

用法:
  python src/visualization/plot_fig7.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import numpy as np

# ───────────────────── 配置 ─────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

METHODS = ["device_only", "cloud_only", "mixed", "janus"]
METHOD_DIRS = {
    "device_only": "device_only",
    "cloud_only": "cloud_only",
    "mixed": "mixed",
    "janus": "janus",
}
METHOD_LABELS = {
    "device_only": "Device-Only",
    "cloud_only": "Cloud-Only",
    "mixed": "Mixed",
    "janus": "Janus",
}

NETWORK_TYPES = ["5g", "lte"]
SCENARIOS = ["static", "walking", "driving"]

# 与参考图片一致的配色
COLORS = {
    "device_only": "#8da0cb",   # 蓝紫色
    "cloud_only": "#fc8d62",    # 橙色
    "mixed": "#a6d854",         # 绿色
    "janus": "#66c2a5",         # 青绿色
}

# 柱状图斜线填充 (与参考图片一致, device_only 有斜线)
HATCHES = {
    "device_only": "//",
    "cloud_only": "",
    "mixed": "",
    "janus": "",
}


# ───────────────────── 数据加载 ─────────────────────

def load_summary(method: str, network: str, scenario: str) -> dict:
    """读取单个 summary CSV，返回字典。"""
    dir_name = METHOD_DIRS[method]
    prefix = method
    fname = f"{prefix}_{network}_{scenario}_summary.csv"
    fpath = os.path.join(RESULTS_DIR, dir_name, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] 文件不存在: {fpath}")
        return None
    with open(fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row  # 只有一行数据
    return None


def collect_data():
    """
    返回嵌套字典:
      data[network_type][method] = {
          "violation_ratio": float (百分比, 0‑100),
          "throughput_fps": float
      }
    对同一网络类型下的 3 个场景取平均值。
    """
    data = {}
    for net in NETWORK_TYPES:
        data[net] = {}
        for method in METHODS:
            vr_list, tp_list = [], []
            for sc in SCENARIOS:
                row = load_summary(method, net, sc)
                if row is None:
                    continue
                vr_list.append(float(row["violation_ratio"]))
                tp_list.append(float(row["throughput_fps"]))
            if vr_list:
                data[net][method] = {
                    "violation_ratio": np.mean(vr_list) * 100,  # 转为百分比
                    "throughput_fps": np.mean(tp_list),
                }
            else:
                data[net][method] = {"violation_ratio": 0, "throughput_fps": 0}
    return data


# ───────────────────── 绘图 ─────────────────────

def plot_one_network(net_key: str, net_data: dict, save_prefix: str):
    """绘制一个网络类型的对比图（左 Violation Ratio，右 Throughput）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 3.2))
    fig.subplots_adjust(wspace=0.45, bottom=0.28, top=0.92, left=0.12, right=0.96)

    x = np.arange(len(METHODS))
    width = 0.6

    # ── 左子图: Violation Ratio (%) ──
    vr_vals = [net_data[m]["violation_ratio"] for m in METHODS]
    bars1 = ax1.bar(
        x, vr_vals, width,
        color=[COLORS[m] for m in METHODS],
        hatch=[HATCHES[m] for m in METHODS],
        edgecolor="white", linewidth=0.6,
    )
    ax1.set_ylabel("Violation Ratio(%)", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [METHOD_LABELS[m] for m in METHODS],
        fontsize=7.5, rotation=30, ha="right",
    )
    ax1.set_ylim(0, max(vr_vals) * 1.15 if max(vr_vals) > 0 else 1)
    ax1.tick_params(axis="y", labelsize=8)

    # ── 右子图: Avg. Throughput (FPS) ──
    tp_vals = [net_data[m]["throughput_fps"] for m in METHODS]
    bars2 = ax2.bar(
        x, tp_vals, width,
        color=[COLORS[m] for m in METHODS],
        hatch=[HATCHES[m] for m in METHODS],
        edgecolor="white", linewidth=0.6,
    )
    ax2.set_ylabel("Avg. Throughput(FPS)", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [METHOD_LABELS[m] for m in METHODS],
        fontsize=7.5, rotation=30, ha="right",
    )
    ax2.set_ylim(0, max(tp_vals) * 1.15 if max(tp_vals) > 0 else 1)
    ax2.tick_params(axis="y", labelsize=8)

    # ── 标题 ──
    net_label = "5G" if net_key == "5g" else "LTE"
    subtitle_idx = "(b)" if net_key == "5g" else "(a)"
    fig.text(
        0.5, 0.02,
        f"{subtitle_idx}  Image recognition on {net_label}.",
        ha="center", fontsize=10,
    )

    # ── 保存 ──
    for ext in ("pdf", "png"):
        out = os.path.join(RESULTS_DIR, f"{save_prefix}.{ext}")
        fig.savefig(out, dpi=300)
        print(f"[OK] 已保存: {out}")
    plt.close(fig)


# ───────────────────── 主入口 ─────────────────────

def main():
    data = collect_data()

    # 打印汇总数据供核验
    print("\n========== 汇总数据 ==========")
    for net in NETWORK_TYPES:
        print(f"\n--- {net.upper()} ---")
        for m in METHODS:
            d = data[net][m]
            print(f"  {METHOD_LABELS[m]:15s}  "
                  f"Violation={d['violation_ratio']:6.2f}%  "
                  f"Throughput={d['throughput_fps']:.4f} FPS")

    # 绘图
    plot_one_network("lte", data["lte"], "fig7_lte")
    plot_one_network("5g", data["5g"], "fig7_5g")

    print("\n全部完成！图片保存在 results/ 目录下。")


if __name__ == "__main__":
    main()
