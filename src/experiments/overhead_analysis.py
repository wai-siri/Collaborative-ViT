"""
开销分析脚本
分析Janus各部分的时间开销
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import json

from schedule.schedule import init, schedule
from schedule.split_inference import device_forward, cloud_forward
from schedule.declining_rate import declining_rate
from runtime.compression import compress_tensor, decompress_tensor


def analyze_overhead(model, image, bandwidth, SLA):
    """
    分析Janus各部分开销
    
    Args:
        model: ViT模型
        image: torch.Tensor, 输入图像
        bandwidth: float, 带宽(bps)
        SLA: float, 延迟要求(ms)
    
    Returns:
        dict: 各部分开销
    """
    device = image.device
    
    # 1. 系统开销（调度器）
    t0 = time.time()
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    D_M = model.pos_embed.size(2)
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits
    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)
    alpha, split_point = schedule(N, x_0, D_M, bits, num_steps, step, bandwidth, SLA)
    system_overhead = (time.time() - t0) * 1000
    
    # 2. 设备端计算
    t0 = time.time()
    with torch.no_grad():
        x_mid = device_forward(model, image, alpha, split_point)
    device_compute = (time.time() - t0) * 1000
    
    # 3. 压缩开销
    t0 = time.time()
    compressed = compress_tensor(x_mid)
    compression_time = (time.time() - t0) * 1000
    
    # 4. 传输开销（模拟）
    data_size = len(compressed) * 8  # bits
    transmission = (data_size / bandwidth) * 1000  # ms
    
    # 5. 解压开销
    t0 = time.time()
    x_recovered = decompress_tensor(compressed, device=device)
    decompression_time = (time.time() - t0) * 1000
    
    # 6. 云端计算（模拟，实际需要在云端测量）
    t0 = time.time()
    with torch.no_grad():
        logits = cloud_forward(model, x_recovered, split_point)
    cloud_compute = (time.time() - t0) * 1000
    
    total = system_overhead + device_compute + compression_time + transmission + decompression_time + cloud_compute
    
    return {
        'system': system_overhead,
        'device_compute': device_compute,
        'compression': compression_time,
        'transmission': transmission,
        'decompression': decompression_time,
        'cloud_compute': cloud_compute,
        'total': total,
        'system_pct': system_overhead / total * 100,
        'device_pct': device_compute / total * 100,
        'compression_pct': compression_time / total * 100,
        'transmission_pct': transmission / total * 100,
        'decompression_pct': decompression_time / total * 100,
        'cloud_pct': cloud_compute / total * 100,
        'alpha': alpha,
        'split_point': split_point,
        'data_size_kb': len(compressed) / 1024
    }


if __name__ == "__main__":
    print("Overhead Analysis")
    print("="*60)
    
    # 加载模型
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}\n")
    
    # 虚拟图像
    image = torch.randn(1, 3, 384, 384, device=device)
    
    # 测试不同网络条件
    test_cases = [
        (50, 300),   # 50 Mbps, 300ms SLA
        (100, 300),  # 100 Mbps, 300ms SLA
        (200, 300),  # 200 Mbps, 300ms SLA
        (300, 300),  # 300 Mbps, 300ms SLA
    ]
    
    all_results = []
    
    for bw_mbps, sla in test_cases:
        bw = bw_mbps * 1e6
        overhead = analyze_overhead(model, image, bw, SLA=sla)
        all_results.append(overhead)
        
        print(f"Bandwidth: {bw_mbps} Mbps, SLA: {sla}ms")
        print(f"  α={overhead['alpha']:.4f}, split={overhead['split_point']}")
        print(f"  Data size: {overhead['data_size_kb']:.2f} KB")
        print(f"  System:        {overhead['system']:.2f}ms ({overhead['system_pct']:.2f}%)")
        print(f"  Device:        {overhead['device_compute']:.2f}ms ({overhead['device_pct']:.2f}%)")
        print(f"  Compression:   {overhead['compression']:.2f}ms ({overhead['compression_pct']:.2f}%)")
        print(f"  Transmission:  {overhead['transmission']:.2f}ms ({overhead['transmission_pct']:.2f}%)")
        print(f"  Decompression: {overhead['decompression']:.2f}ms ({overhead['decompression_pct']:.2f}%)")
        print(f"  Cloud:         {overhead['cloud_compute']:.2f}ms ({overhead['cloud_pct']:.2f}%)")
        print(f"  Total:         {overhead['total']:.2f}ms")
        print()
    
    # 保存结果
    with open('overhead_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("="*60)
    print("Results saved to: overhead_analysis.json")
