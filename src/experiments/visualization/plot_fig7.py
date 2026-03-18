"""
Fig. 7可视化 - 整体性能对比
生成论文Fig. 7的复现图表
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_results(scenario, network_type):
    """加载指定场景的结果"""
    filepath = f'results/fig7/{scenario}_{network_type}_summary.json'
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_fig7(network_type='lte', output_path=None):
    """
    绘制Fig. 7图表
    
    Args:
        network_type: str, 'lte' or '5g'
        output_path: str, 输出文件路径
    """
    scenarios = ['static', 'walking', 'driving']
    methods = ['device_only', 'cloud_only', 'mixed', 'janus']
    method_labels = ['Device-Only', 'Cloud-Only', 'Mixed', 'Janus']
    
    # 加载所有场景的结果
    all_results = {}
    for scenario in scenarios:
        results = load_results(scenario, network_type)
        if results:
            all_results[scenario] = results
    
    if not all_results:
        print(f"Error: No results found for {network_type}")
        return
    
    # 准备数据
    throughputs = {method: [] for method in methods}
    violations = {method: [] for method in methods}
    
    for scenario in scenarios:
        if scenario not in all_results:
            for method in methods:
                throughputs[method].append(0)
                violations[method].append(0)
            continue
        
        for method in methods:
            if method in all_results[scenario]:
                throughputs[method].append(all_results[scenario][method]['throughput'])
                violations[method].append(all_results[scenario][method]['violation_ratio'])
            else:
                throughputs[method].append(0)
                violations[method].append(0)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 子图1：Throughput
    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
        ax1.bar(x + offset, throughputs[method], width, 
                label=method_labels[i], color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('Throughput (FPS)', fontsize=12)
    ax1.set_title(f'{network_type.upper()} - Throughput', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Static', 'Walking', 'Driving'])
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 子图2：Violation Ratio
    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
        ax2.bar(x + offset, violations[method], width, 
                label=method_labels[i], color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('Violation Ratio (%)', fontsize=12)
    ax2.set_title(f'{network_type.upper()} - Latency Violation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Static', 'Walking', 'Driving'])
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        output_path = f'figures/fig7_{network_type}_performance.png'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()


def plot_combined_fig7(output_path='figures/fig7_combined_performance.png'):
    """
    绘制LTE和5G的组合图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = ['static', 'walking', 'driving']
    methods = ['device_only', 'cloud_only', 'mixed', 'janus']
    method_labels = ['Device-Only', 'Cloud-Only', 'Mixed', 'Janus']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for row, network_type in enumerate(['lte', '5g']):
        # 加载结果
        all_results = {}
        for scenario in scenarios:
            results = load_results(scenario, network_type)
            if results:
                all_results[scenario] = results
        
        if not all_results:
            continue
        
        # 准备数据
        throughputs = {method: [] for method in methods}
        violations = {method: [] for method in methods}
        
        for scenario in scenarios:
            if scenario not in all_results:
                for method in methods:
                    throughputs[method].append(0)
                    violations[method].append(0)
                continue
            
            for method in methods:
                if method in all_results[scenario]:
                    throughputs[method].append(all_results[scenario][method]['throughput'])
                    violations[method].append(all_results[scenario][method]['violation_ratio'])
                else:
                    throughputs[method].append(0)
                    violations[method].append(0)
        
        # Throughput
        ax = axes[row, 0]
        for i, method in enumerate(methods):
            offset = (i - 1.5) * width
            ax.bar(x + offset, throughputs[method], width, 
                   label=method_labels[i], color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Throughput (FPS)', fontsize=12)
        ax.set_title(f'{network_type.upper()} - Throughput', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Static', 'Walking', 'Driving'])
        if row == 0:
            ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Violation Ratio
        ax = axes[row, 1]
        for i, method in enumerate(methods):
            offset = (i - 1.5) * width
            ax.bar(x + offset, violations[method], width, 
                   label=method_labels[i], color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Violation Ratio (%)', fontsize=12)
        ax.set_title(f'{network_type.upper()} - Latency Violation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Static', 'Walking', 'Driving'])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Fig. 7 results')
    parser.add_argument('--network', type=str, choices=['lte', '5g', 'both'], 
                       default='both', help='Network type to plot')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    args = parser.parse_args()
    
    print("="*60)
    print("Plotting Fig. 7 - Overall Performance")
    print("="*60)
    
    if args.network == 'both':
        plot_combined_fig7(args.output if args.output else 'figures/fig7_combined_performance.png')
    else:
        plot_fig7(args.network, args.output)
    
    print("="*60)
    print("✓ Plotting completed!")
    print("="*60)


if __name__ == "__main__":
    main()
