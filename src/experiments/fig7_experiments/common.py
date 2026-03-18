"""
Fig. 7实验的共享函数模块
提供通用的实验运行、结果保存等功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd
import json
import time
from tqdm import tqdm

from schedule.schedule import init
from runtime.jdevice import run_jdevice
from experiments.baselines import device_only, cloud_only, mixed
from experiments.evaluate import compute_metrics


def load_trace(scenario, network_type, trace_dir=None):
    """
    加载指定场景的网络trace
    
    Args:
        scenario: str, 场景名称 (static, walking, driving)
        network_type: str, 网络类型 (5g, lte)
        trace_dir: str, trace文件目录
    
    Returns:
        pd.DataFrame: trace数据
    """
    # 如果未指定 trace_dir，使用项目根目录下的 assets/network_traces
    if trace_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/experiments/fig7_experiments/ 向上三级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        trace_dir = os.path.join(project_root, 'assets', 'network_traces')
    
    trace_path = os.path.join(trace_dir, f'{scenario}_{network_type}_trace.csv')
    
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    
    df = pd.read_csv(trace_path)
    print(f"Loaded trace: {trace_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Bandwidth - Mean: {df['bandwidth_mbps'].mean():.2f} Mbps, "
          f"Std: {df['bandwidth_mbps'].std():.2f} Mbps")
    
    return df


def run_single_method(method_name, model, loader, trace_df, SLA, server_ip='127.0.0.1', server_port=9999):
    """
    运行单个方法的完整实验
    
    Args:
        method_name: str, 方法名称 (janus, device_only, cloud_only, mixed)
        model: ViT模型
        loader: DataLoader
        trace_df: pd.DataFrame, 网络trace数据
        SLA: float, 延迟要求(ms)
        server_ip: str, 云服务器IP
        server_port: int, 云服务器端口
    
    Returns:
        dict: 实验结果
    """
    device = next(model.parameters()).device
    
    latencies = []
    predictions = []
    labels = []
    alphas = []
    splits = []
    bandwidths = []
    
    trace_idx = 0
    total_samples = len(loader.dataset)
    
    print(f"\n{'='*60}")
    print(f"Running {method_name.upper()}")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"SLA: {SLA} ms")
    
    for images, targets in tqdm(loader, desc=f"{method_name}"):
        images = images.to(device)
        
        # 从trace中获取当前带宽
        bw_mbps = trace_df.iloc[trace_idx % len(trace_df)]['bandwidth_mbps']
        bw_bps = bw_mbps * 1e6  # 转换为bps
        bandwidths.append(bw_mbps)
        trace_idx += 1
        
        try:
            if method_name == 'janus':
                # Janus方法需要连接云服务器
                logits, lat, alpha, split = run_jdevice(
                    images, model, server_ip, server_port, SLA, [bw_bps]
                )
                alphas.append(alpha)
                splits.append(split)
                
            elif method_name == 'device_only':
                # Device-Only方法
                logits, lat = device_only(model, images)
                
            elif method_name == 'cloud_only':
                # Cloud-Only方法需要连接云服务器
                logits, lat = cloud_only(model, images, server_ip, server_port)
                
            elif method_name == 'mixed':
                # Mixed方法需要连接云服务器
                logits, lat = mixed(model, images, server_ip, server_port, bw_bps, SLA)
            
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # 记录结果
            latencies.append(lat)
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
            labels.append(targets.item())
            
        except Exception as e:
            print(f"\nError on sample {len(latencies)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算评估指标
    metrics = compute_metrics(latencies, predictions, labels, SLA)
    
    # 添加额外信息
    metrics['method'] = method_name
    metrics['total_samples'] = len(latencies)
    metrics['avg_bandwidth_mbps'] = sum(bandwidths) / len(bandwidths) if bandwidths else 0
    
    if alphas:
        metrics['avg_alpha'] = sum(alphas) / len(alphas)
        metrics['avg_split'] = sum(splits) / len(splits)
    
    # 打印结果摘要
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} Results:")
    print(f"{'='*60}")
    print(f"Samples: {metrics['total_samples']}")
    print(f"Avg Latency: {metrics['avg_latency']:.2f} ms")
    print(f"Violation Ratio: {metrics['violation_ratio']:.2f}%")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Throughput: {metrics['throughput']:.2f} FPS")
    if alphas:
        print(f"Avg Alpha: {metrics['avg_alpha']:.4f}")
        print(f"Avg Split: {metrics['avg_split']:.1f}")
    print(f"{'='*60}\n")
    
    # 返回详细结果
    return {
        'metrics': metrics,
        'latencies': latencies,
        'predictions': predictions,
        'labels': labels,
        'alphas': alphas if alphas else None,
        'splits': splits if splits else None,
        'bandwidths': bandwidths
    }


def save_experiment_results(results, output_path):
    """
    保存实验结果到JSON文件
    
    Args:
        results: dict, 实验结果
        output_path: str, 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def save_summary(all_results, output_path):
    """
    保存所有方法的汇总结果
    
    Args:
        all_results: dict, 所有方法的结果
        output_path: str, 输出文件路径
    """
    summary = {}
    
    for method, result in all_results.items():
        summary[method] = result['metrics']
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_path}")
    
    # 打印对比表格
    print("\n" + "="*80)
    print("SUMMARY - All Methods Comparison")
    print("="*80)
    print(f"{'Method':<15} {'Latency(ms)':<15} {'Violation(%)':<15} {'Accuracy(%)':<15} {'Throughput(FPS)':<15}")
    print("-"*80)
    
    for method, metrics in summary.items():
        print(f"{method:<15} "
              f"{metrics['avg_latency']:<15.2f} "
              f"{metrics['violation_ratio']:<15.2f} "
              f"{metrics['accuracy']:<15.2f} "
              f"{metrics['throughput']:<15.2f}")
    
    print("="*80 + "\n")
