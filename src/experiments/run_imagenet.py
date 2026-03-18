"""
ImageNet实验脚本
运行Janus和所有基线方法的完整实验
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import argparse

from schedule.schedule import init
from runtime.jdevice import run_jdevice
from experiments.baselines import device_only, cloud_only, mixed
from experiments.evaluate import compute_metrics, save_results, plot_comparison, plot_latency_distribution


# 实验参数
DATA_DIR = "data/imagenet/val"
MODEL_NAME = "vit_large_patch16_384"
SLA = 300  # ms
SERVER_IP = "127.0.0.1"  # 通过SSH隧道连接
SERVER_PORT = 9999  # SSH隧道映射的端口
NUM_SAMPLES = 100
BATCH_SIZE = 1


def get_imagenet_loader(data_dir, num_samples=None):
    """
    加载ImageNet数据
    
    Args:
        data_dir: str, 数据集路径
        num_samples: int, 样本数量（None表示全部）
    
    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Warning: ImageNet dataset not found at {data_dir}")
        print("Using random data for testing...")
        return None
    
    if num_samples:
        indices = list(range(min(num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    return loader


def run_experiment(method_name, model, loader, device, server_ip, server_port, SLA):
    """
    运行单个方法的实验
    
    Args:
        method_name: str, 方法名称
        model: ViT模型
        loader: DataLoader
        device: torch.device
        server_ip: str, 服务器IP
        server_port: int, 服务器端口
        SLA: float, 延迟要求(ms)
    
    Returns:
        dict: 实验结果
    """
    latencies = []
    predictions = []
    labels = []
    alphas = []
    splits = []
    
    bandwidth_history = []
    
    print(f"\nRunning {method_name}...")
    
    if loader is None:
        # 使用随机数据测试
        print("Using random data for testing...")
        for i in tqdm(range(10)):
            images = torch.randn(1, 3, 384, 384, device=device)
            targets = torch.tensor([i % 1000])
            
            try:
                if method_name == "janus":
                    logits, lat, alpha, split = run_jdevice(
                        images, model, server_ip, server_port, SLA, bandwidth_history
                    )
                    alphas.append(alpha)
                    splits.append(split)
                    
                elif method_name == "device_only":
                    logits, lat = device_only(model, images)
                    
                elif method_name == "cloud_only":
                    logits, lat = cloud_only(model, images, server_ip, server_port)
                    
                elif method_name == "mixed":
                    bw = bandwidth_history[-1] if bandwidth_history else 100e6
                    logits, lat = mixed(model, images, server_ip, server_port, bw, SLA)
                
                latencies.append(lat)
                pred = torch.argmax(logits, dim=-1).item()
                predictions.append(pred)
                labels.append(targets.item())
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        # 使用真实数据
        for images, targets in tqdm(loader):
            images = images.to(device)
            
            try:
                if method_name == "janus":
                    logits, lat, alpha, split = run_jdevice(
                        images, model, server_ip, server_port, SLA, bandwidth_history
                    )
                    alphas.append(alpha)
                    splits.append(split)
                    
                elif method_name == "device_only":
                    logits, lat = device_only(model, images)
                    
                elif method_name == "cloud_only":
                    logits, lat = cloud_only(model, images, server_ip, server_port)
                    
                elif method_name == "mixed":
                    bw = bandwidth_history[-1] if bandwidth_history else 100e6
                    logits, lat = mixed(model, images, server_ip, server_port, bw, SLA)
                
                latencies.append(lat)
                pred = torch.argmax(logits, dim=-1).item()
                predictions.append(pred)
                labels.append(targets.item())
                
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # 计算指标
    metrics = compute_metrics(latencies, predictions, labels, SLA)
    
    if alphas:
        metrics['avg_alpha'] = sum(alphas) / len(alphas)
        metrics['avg_split'] = sum(splits) / len(splits)
    
    return metrics, latencies


def main():
    parser = argparse.ArgumentParser(description='Run ImageNet experiments')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='ImageNet validation set path')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help='Number of samples to test')
    parser.add_argument('--sla', type=float, default=SLA, help='Latency SLA in ms')
    parser.add_argument('--server_ip', type=str, default=SERVER_IP, help='Cloud server IP')
    parser.add_argument('--server_port', type=int, default=SERVER_PORT, help='Cloud server port')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['janus', 'device_only', 'cloud_only', 'mixed'],
                       help='Methods to test')
    args = parser.parse_args()
    
    # 加载模型
    print("Loading model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # 加载数据
    print(f"\nLoading data from {args.data_dir}...")
    loader = get_imagenet_loader(args.data_dir, args.num_samples)
    
    # 运行所有方法
    results = {}
    all_latencies = {}
    
    for method in args.methods:
        try:
            metrics, latencies = run_experiment(
                method, model, loader, device, 
                args.server_ip, args.server_port, args.sla
            )
            results[method] = metrics
            all_latencies[method] = latencies
            
            print(f"\n{method} results:")
            print(json.dumps(metrics, indent=2))
            
        except Exception as e:
            print(f"\nError running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    print("\nSaving results...")
    save_results(results, "results_imagenet.json")
    
    # 绘图
    print("Generating plots...")
    plot_comparison(results, "comparison_imagenet.png")
    plot_latency_distribution(all_latencies, args.sla, "latency_distribution.png")
    
    print("\n" + "="*50)
    print("Experiment completed!")
    print("="*50)
    print(f"Results saved to: results_imagenet.json")
    print(f"Plots saved to: comparison_imagenet.png, latency_distribution.png")


if __name__ == "__main__":
    main()
