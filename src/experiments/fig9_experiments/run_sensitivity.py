"""
Fig. 9实验 - 带宽敏感性分析
扫描不同带宽下Janus和基线方法的性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pandas as pd
import json
import argparse
from tqdm import tqdm

from utils.imagenet_loader import get_imagenet_loader
from schedule.schedule import init
from runtime.jdevice import run_jdevice
from experiments.baselines import device_only, cloud_only


def run_sensitivity_analysis(model, loader, bandwidths, SLA, server_ip='127.0.0.1', server_port=9999, num_samples=100):
    """
    运行带宽敏感性分析
    
    Args:
        model: ViT模型
        loader: DataLoader
        bandwidths: list, 带宽列表（Mbps）
        SLA: float, 延迟要求(ms)
        server_ip: str, 云服务器IP
        server_port: int, 云服务器端口
        num_samples: int, 每个带宽测试的样本数
    
    Returns:
        dict: 分析结果
    """
    device = next(model.parameters()).device
    
    results = {
        'bandwidths': bandwidths,
        'janus': {
            'latencies': [],
            'alphas': [],
            'splits': [],
            'violations': []
        },
        'device_only': {
            'latencies': [],
            'violations': []
        },
        'cloud_only': {
            'latencies': [],
            'violations': []
        }
    }
    
    # 准备测试数据
    test_data = []
    for i, (images, labels) in enumerate(loader):
        if i >= num_samples:
            break
        test_data.append((images.to(device), labels))
    
    print(f"Prepared {len(test_data)} test samples\n")
    
    # 扫描带宽
    for bw_mbps in bandwidths:
        bw_bps = bw_mbps * 1e6
        
        print(f"\n{'='*60}")
        print(f"Testing Bandwidth: {bw_mbps} Mbps")
        print(f"{'='*60}")
        
        # Janus方法
        print("\nTesting Janus...")
        janus_lats = []
        janus_alphas = []
        janus_splits = []
        
        for images, _ in tqdm(test_data, desc="Janus"):
            try:
                logits, lat, alpha, split = run_jdevice(
                    images, model, server_ip, server_port, SLA, [bw_bps]
                )
                janus_lats.append(lat)
                janus_alphas.append(alpha)
                janus_splits.append(split)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        avg_lat = sum(janus_lats) / len(janus_lats) if janus_lats else 0
        avg_alpha = sum(janus_alphas) / len(janus_alphas) if janus_alphas else 0
        avg_split = sum(janus_splits) / len(janus_splits) if janus_splits else 0
        violation = sum(1 for lat in janus_lats if lat > SLA) / len(janus_lats) * 100 if janus_lats else 0
        
        results['janus']['latencies'].append(avg_lat)
        results['janus']['alphas'].append(avg_alpha)
        results['janus']['splits'].append(avg_split)
        results['janus']['violations'].append(violation)
        
        print(f"Janus - Latency: {avg_lat:.2f} ms, Alpha: {avg_alpha:.4f}, Split: {avg_split:.1f}, Violation: {violation:.2f}%")
        
        # Device-Only方法
        print("\nTesting Device-Only...")
        device_lats = []
        
        for images, _ in tqdm(test_data, desc="Device-Only"):
            try:
                logits, lat = device_only(model, images)
                device_lats.append(lat)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        avg_lat = sum(device_lats) / len(device_lats) if device_lats else 0
        violation = sum(1 for lat in device_lats if lat > SLA) / len(device_lats) * 100 if device_lats else 0
        
        results['device_only']['latencies'].append(avg_lat)
        results['device_only']['violations'].append(violation)
        
        print(f"Device-Only - Latency: {avg_lat:.2f} ms, Violation: {violation:.2f}%")
        
        # Cloud-Only方法
        print("\nTesting Cloud-Only...")
        cloud_lats = []
        
        for images, _ in tqdm(test_data, desc="Cloud-Only"):
            try:
                logits, lat = cloud_only(model, images, server_ip, server_port)
                cloud_lats.append(lat)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        avg_lat = sum(cloud_lats) / len(cloud_lats) if cloud_lats else 0
        violation = sum(1 for lat in cloud_lats if lat > SLA) / len(cloud_lats) * 100 if cloud_lats else 0
        
        results['cloud_only']['latencies'].append(avg_lat)
        results['cloud_only']['violations'].append(violation)
        
        print(f"Cloud-Only - Latency: {avg_lat:.2f} ms, Violation: {violation:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run Fig. 9 Sensitivity Analysis')
    parser.add_argument('--data_path', type=str, 
                       default='data/validation-00000-of-00014.parquet',
                       help='ImageNet parquet file path')
    parser.add_argument('--sla', type=float, default=300, 
                       help='Latency SLA in ms')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1',
                       help='Cloud server IP (for SSH tunnel)')
    parser.add_argument('--server_port', type=int, default=9999,
                       help='Cloud server port')
    parser.add_argument('--bw_min', type=int, default=5,
                       help='Minimum bandwidth (Mbps)')
    parser.add_argument('--bw_max', type=int, default=50,
                       help='Maximum bandwidth (Mbps)')
    parser.add_argument('--bw_step', type=int, default=5,
                       help='Bandwidth step (Mbps)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per bandwidth')
    args = parser.parse_args()
    
    print("="*80)
    print("Fig. 9 Experiment - Bandwidth Sensitivity Analysis")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"SLA: {args.sla} ms")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Bandwidth range: {args.bw_min}-{args.bw_max} Mbps (step {args.bw_step})")
    print(f"Samples per bandwidth: {args.num_samples}")
    print("="*80 + "\n")
    
    # 生成带宽列表
    bandwidths = list(range(args.bw_min, args.bw_max + 1, args.bw_step))
    print(f"Testing bandwidths: {bandwidths} Mbps\n")
    
    # 加载模型
    print("Loading ViT model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    
    # 加载ImageNet数据
    print("Loading ImageNet data...")
    loader = get_imagenet_loader(args.data_path, batch_size=1)
    print(f"Loaded {len(loader.dataset)} images\n")
    
    print("⚠️  This experiment requires cloud server connection!")
    print("   Please ensure Jcloud server is running on AutoDL")
    print(f"   and SSH tunnel is established: ssh -L {args.server_port}:localhost:9999 autodl")
    input("   Press Enter to continue...")
    
    # 运行敏感性分析
    results = run_sensitivity_analysis(
        model, loader, bandwidths, args.sla,
        args.server_ip, args.server_port, args.num_samples
    )
    
    # 保存结果
    output_path = 'results/fig9/sensitivity_analysis.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Sensitivity analysis completed!")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print("="*80)
    
    # 打印摘要
    print("\nSummary:")
    print(f"{'Bandwidth(Mbps)':<15} {'Janus(ms)':<15} {'Device(ms)':<15} {'Cloud(ms)':<15}")
    print("-"*60)
    for i, bw in enumerate(bandwidths):
        print(f"{bw:<15} "
              f"{results['janus']['latencies'][i]:<15.2f} "
              f"{results['device_only']['latencies'][i]:<15.2f} "
              f"{results['cloud_only']['latencies'][i]:<15.2f}")


if __name__ == "__main__":
    main()
