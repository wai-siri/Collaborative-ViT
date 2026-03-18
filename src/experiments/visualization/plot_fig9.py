"""
Fig. 9可视化 - 带宽敏感性分析
生成论文Fig. 9的复现图表
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_fig9(results_path='results/fig9/sensitivity_analysis.json', output_path='figures/fig9_sensitivity.png'):
    """
    绘制Fig. 9图表
    
    Args:
        results_path: str, 结果文件路径
        output_path: str, 输出文件路径
    """
    # 加载结果
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    bandwidths = results['bandwidths']
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 颜色方案
    colors = {
        'janus': '#d62728',
        'device_only': '#1f77b4',
        'cloud_only': '#ff7f0e'
    }
    
    labels = {
        'janus': 'Janus',
        'device_only': 'Device-Only',
        'cloud_only': 'Cloud-Only'
    }
    
    # 子图1：延迟 vs 带宽
    ax1.plot(bandwidths, results['janus']['latencies'], 
             marker='o', linewidth=2, color=colors['janus'], label=labels['janus'])
    ax1.plot(bandwidths, results['device_only']['latencies'], 
             marker='s', linewidth=2, color=colors['device_only'], label=labels['device_only'])
    ax1.plot(bandwidths, results['cloud_only']['latencies'], 
             marker='^', linewidth=2, color=colors['cloud_only'], label=labels['cloud_only'])
    
    ax1.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency vs Bandwidth', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：违规率 vs 带宽
    ax2.plot(bandwidths, results['janus']['violations'], 
             marker='o', linewidth=2, color=colors['janus'], label=labels['janus'])
    ax2.plot(bandwidths, results['device_only']['violations'], 
             marker='s', linewidth=2, color=colors['device_only'], label=labels['device_only'])
    ax2.plot(bandwidths, results['cloud_only']['violations'], 
             marker='^', linewidth=2, color=colors['cloud_only'], label=labels['cloud_only'])
    
    ax2.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax2.set_ylabel('Violation Ratio (%)', fontsize=12)
    ax2.set_title('Violation Ratio vs Bandwidth', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 子图3：Alpha vs 带宽
    ax3.plot(bandwidths, results['janus']['alphas'], 
             marker='o', linewidth=2, color=colors['janus'])
    
    ax3.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax3.set_ylabel('Declining Rate (α)', fontsize=12)
    ax3.set_title('Janus - Declining Rate vs Bandwidth', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 子图4：Split Point vs 带宽
    ax4.plot(bandwidths, results['janus']['splits'], 
             marker='o', linewidth=2, color=colors['janus'])
    
    ax4.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax4.set_ylabel('Split Point (Layer)', fontsize=12)
    ax4.set_title('Janus - Split Point vs Bandwidth', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Fig. 9 results')
    parser.add_argument('--input', type=str, 
                       default='results/fig9/sensitivity_analysis.json',
                       help='Input results file')
    parser.add_argument('--output', type=str, 
                       default='figures/fig9_sensitivity.png',
                       help='Output file path')
    args = parser.parse_args()
    
    print("="*60)
    print("Plotting Fig. 9 - Bandwidth Sensitivity Analysis")
    print("="*60)
    
    plot_fig9(args.input, args.output)
    
    print("="*60)
    print("✓ Plotting completed!")
    print("="*60)


if __name__ == "__main__":
    main()
