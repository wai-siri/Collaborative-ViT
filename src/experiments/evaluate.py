"""
评估指标模块
实现论文中的所有评估指标
"""

import numpy as np
import json
import matplotlib.pyplot as plt


def compute_metrics(latencies, predictions, labels, SLA):
    """
    计算所有评估指标
    
    Args:
        latencies: list of float, 每个样本的延迟(ms)
        predictions: list of int, 预测类别
        labels: list of int, 真实类别
        SLA: float, 延迟要求(ms)
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 1. Violation Ratio
    violations = sum(1 for lat in latencies if lat > SLA)
    violation_ratio = violations / len(latencies) if latencies else 0
    
    # 2. Accuracy
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0
    
    # 3. Throughput (FPS)
    avg_latency = np.mean(latencies) if latencies else float('inf')
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0
    
    # 4. Latency Deviation Rate
    deviations = [max(0, (lat - SLA) / SLA) for lat in latencies]
    avg_deviation = np.mean(deviations) if deviations else 0
    
    return {
        'violation_ratio': violation_ratio,
        'accuracy': accuracy,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'latency_deviation': avg_deviation,
        'num_samples': len(latencies)
    }


def save_results(results, filename):
    """
    保存结果到JSON
    
    Args:
        results: dict, 结果字典
        filename: str, 保存路径
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {filename}")


def plot_comparison(results_dict, save_path):
    """
    绘制对比图
    
    Args:
        results_dict: dict, {method_name: metrics_dict}
        save_path: str, 保存路径
    """
    methods = list(results_dict.keys())
    
    # 提取指标
    throughputs = [results_dict[m]['throughput'] for m in methods]
    violations = [results_dict[m]['violation_ratio'] * 100 for m in methods]
    accuracies = [results_dict[m]['accuracy'] * 100 for m in methods]
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Throughput
    axes[0].bar(methods, throughputs, color='steelblue')
    axes[0].set_ylabel('Throughput (FPS)')
    axes[0].set_title('Average Throughput')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Violation Ratio
    axes[1].bar(methods, violations, color='coral')
    axes[1].set_ylabel('Violation Ratio (%)')
    axes[1].set_title('Latency Violation Ratio')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Accuracy
    axes[2].bar(methods, accuracies, color='seagreen')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Inference Accuracy')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_latency_distribution(latencies_dict, SLA, save_path):
    """
    绘制延迟分布图
    
    Args:
        latencies_dict: dict, {method_name: [latencies]}
        SLA: float, 延迟要求
        save_path: str, 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    for method, latencies in latencies_dict.items():
        plt.hist(latencies, bins=50, alpha=0.5, label=method)
    
    plt.axvline(x=SLA, color='red', linestyle='--', linewidth=2, label=f'SLA={SLA}ms')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Latency distribution plot saved to: {save_path}")


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    # 模拟数据
    latencies = [50, 60, 120, 80, 90, 110, 70, 85, 95, 105]
    predictions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [1, 2, 4, 4, 5, 6, 7, 8, 10, 10]
    SLA = 100
    
    # 计算指标
    metrics = compute_metrics(latencies, predictions, labels, SLA)
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试保存
    save_results({'test_method': metrics}, 'test_results.json')
    
    # 测试对比图
    results_dict = {
        'Janus': {'throughput': 15.5, 'violation_ratio': 0.05, 'accuracy': 0.95},
        'Device-Only': {'throughput': 12.0, 'violation_ratio': 0.02, 'accuracy': 0.92},
        'Cloud-Only': {'throughput': 10.0, 'violation_ratio': 0.15, 'accuracy': 0.96},
    }
    plot_comparison(results_dict, 'test_comparison.png')
    
    # 测试延迟分布图
    latencies_dict = {
        'Janus': [50, 60, 70, 80, 90],
        'Device-Only': [40, 50, 60, 70, 80],
        'Cloud-Only': [100, 110, 120, 130, 140],
    }
    plot_latency_distribution(latencies_dict, SLA, 'test_latency_dist.png')
    
    print("\nEvaluation module tests completed!")
