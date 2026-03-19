"""
网络trace数据解析工具
从真实的.list文件解析生成CSV格式的网络trace数据
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


def parse_list_file(filepath):
    """
    解析单个.list文件
    
    Args:
        filepath: str, .list文件路径
    
    Returns:
        list: 带宽值列表（Mbps）
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 按Run分割
    runs = content.split('###############################')
    
    all_values = []
    for run in runs:
        run = run.strip()
        if not run:
            continue
        
        # 提取数据行（跳过Run X标题）
        lines = run.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Run'):
                continue
            
            # 逗号分隔的带宽值（可能很长，被截断显示）
            try:
                values = [float(v.strip()) for v in line.split(',') if v.strip()]
                all_values.extend(values)
            except ValueError as e:
                # 跳过无法解析的行
                continue
    
    return all_values


def merge_list_files(list_files):
    """
    合并多个.list文件的数据
    
    Args:
        list_files: list, .list文件路径列表
    
    Returns:
        list: 合并后的带宽值列表
    """
    all_values = []
    for filepath in list_files:
        if os.path.exists(filepath):
            values = parse_list_file(filepath)
            all_values.extend(values)
            print(f"Parsed {filepath}: {len(values)} samples")
        else:
            print(f"Warning: {filepath} not found")
    
    return all_values


def generate_trace_csv(scenario, network_type, base_dir=None, output_dir=None):
    """
    生成特定场景的trace CSV文件
    
    Args:
        scenario: str, 场景名称 (static, walking, driving)
        network_type: str, 网络类型 (5g, lte)
        base_dir: str, 原始数据目录
        output_dir: str, 输出目录
    
    Returns:
        str: 输出文件路径
    """
    # 如果未指定路径，使用项目根目录下的 assets/network_traces
    if base_dir is None or output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/utils/ 向上两级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(script_dir))
        default_dir = os.path.join(project_root, 'assets', 'network_traces')
        if base_dir is None:
            base_dir = default_dir
        if output_dir is None:
            output_dir = default_dir
    
    # 构建输入路径
    throughput_dir = os.path.join(base_dir, 'throughput', scenario, network_type)
    
    # 获取所有.list文件
    list_files = []
    if os.path.exists(throughput_dir):
        for filename in os.listdir(throughput_dir):
            if filename.endswith('.list'):
                list_files.append(os.path.join(throughput_dir, filename))
    
    if not list_files:
        print(f"Warning: No .list files found in {throughput_dir}")
        return None
    
    print(f"\nGenerating trace for {scenario} {network_type}...")
    print(f"Found {len(list_files)} .list files")
    
    # 解析并合并数据
    bandwidth_values = merge_list_files(list_files)
    
    if not bandwidth_values:
        print(f"Error: No data parsed from {throughput_dir}")
        return None
    
    # 生成时间戳（假设采样率10 samples/s）
    sampling_rate = 10  # Hz
    timestamps = np.arange(len(bandwidth_values)) / sampling_rate
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bandwidth_mbps': bandwidth_values,
        'rtt_ms': 20.0  # 默认RTT，可以后续从ping数据补充
    })
    
    # 保存CSV
    output_filename = f"{scenario}_{network_type}_trace.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    print(f"Bandwidth - Mean: {df['bandwidth_mbps'].mean():.2f} Mbps, "
          f"Std: {df['bandwidth_mbps'].std():.2f} Mbps, "
          f"Min: {df['bandwidth_mbps'].min():.2f} Mbps, "
          f"Max: {df['bandwidth_mbps'].max():.2f} Mbps")
    
    return output_path


def generate_all_traces(base_dir=None, output_dir=None):
    """
    生成所有6个场景的trace CSV文件
    
    Args:
        base_dir: str, 原始数据目录
        output_dir: str, 输出目录
    """
    scenarios = ['static', 'walking', 'driving']
    network_types = ['5g', 'lte']
    
    print("="*60)
    print("Parsing Network Trace Data")
    print("="*60)
    
    generated_files = []
    for scenario in scenarios:
        for network_type in network_types:
            output_path = generate_trace_csv(scenario, network_type, base_dir, output_dir)
            if output_path:
                generated_files.append(output_path)
    
    print("\n" + "="*60)
    print(f"Successfully generated {len(generated_files)} trace files:")
    for filepath in generated_files:
        print(f"  - {filepath}")
    print("="*60)
    
    return generated_files


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    
    # 生成所有trace文件
    generate_all_traces()
