"""
基线方法实现
包括：Device-Only, Cloud-Only, Mixed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from schedule.split_inference import device_forward, cloud_forward
from runtime.compression import compress_tensor, decompress_tensor
from runtime.communication import DeviceClient


def device_only(model, image, alpha_fixed=0.23):
    """
    Device-Only基线：固定剪枝率，仅设备端推理
    
    Args:
        model: ViT模型
        image: torch.Tensor, 输入图像
        alpha_fixed: float, 固定剪枝率
    
    Returns:
        logits: torch.Tensor, 推理结果
        latency: float, 延迟(ms)
    """
    start_time = time.time()
    
    N = len(model.blocks)
    
    # 固定剪枝率，完整推理
    with torch.no_grad():
        x_mid = device_forward(model, image, alpha_fixed, N)
        logits = cloud_forward(model, x_mid, N)
    
    latency = (time.time() - start_time) * 1000
    return logits, latency


def cloud_only(model, image, server_ip, server_port):
    """
    Cloud-Only基线：压缩图像，全部云端推理
    
    Args:
        model: ViT模型
        image: torch.Tensor, 输入图像
        server_ip: str, 云端服务器IP
        server_port: int, 云端服务器端口
    
    Returns:
        logits: torch.Tensor, 推理结果
        latency: float, 延迟(ms)
    """
    start_time = time.time()
    
    # 压缩原始图像
    compressed = compress_tensor(image)
    
    # 发送到云端
    client = DeviceClient(server_ip, server_port)
    client.connect()
    client.send_data(compressed, 0.0, 0)  # α=0, split=0表示全云端
    result_data = client.receive_result()
    client.close()
    
    # 解压结果
    logits = decompress_tensor(result_data, device=image.device)
    
    latency = (time.time() - start_time) * 1000
    return logits, latency


def mixed(model, image, server_ip, server_port, bandwidth, SLA):
    """
    Mixed基线：根据网络选择Device-Only或Cloud-Only
    
    Args:
        model: ViT模型
        image: torch.Tensor, 输入图像
        server_ip: str, 云端服务器IP
        server_port: int, 云端服务器端口
        bandwidth: float, 当前带宽(bps)
        SLA: float, 延迟要求(ms)
    
    Returns:
        logits: torch.Tensor, 推理结果
        latency: float, 延迟(ms)
    """
    # 简单策略：如果带宽高，用Cloud-Only，否则用Device-Only
    threshold_bandwidth = 50e6  # 50 Mbps
    
    if bandwidth > threshold_bandwidth:
        return cloud_only(model, image, server_ip, server_port)
    else:
        return device_only(model, image)


if __name__ == "__main__":
    from schedule.schedule import init
    
    print("Testing baseline methods...")
    
    # 加载模型
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 虚拟图像
    image = torch.randn(1, 3, 384, 384, device=device)
    
    # 测试Device-Only
    print("\n1. Testing Device-Only...")
    logits, lat = device_only(model, image)
    print(f"   Latency: {lat:.2f}ms")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Top-5 classes: {torch.topk(logits, 5, dim=-1).indices.tolist()}")
    
    # 测试Cloud-Only（需要启动服务器）
    print("\n2. Testing Cloud-Only...")
    print("   Note: This requires Jcloud server running")
    try:
        logits, lat = cloud_only(model, image, '127.0.0.1', 9999)
        print(f"   Latency: {lat:.2f}ms")
        print(f"   Logits shape: {logits.shape}")
    except ConnectionRefusedError:
        print("   Skipped: Server not running")
    
    # 测试Mixed
    print("\n3. Testing Mixed...")
    print("   Note: This requires Jcloud server running")
    try:
        logits, lat = mixed(model, image, '127.0.0.1', 9999, 100e6, 100.0)
        print(f"   Latency: {lat:.2f}ms")
        print(f"   Logits shape: {logits.shape}")
    except ConnectionRefusedError:
        print("   Skipped: Server not running")
    
    print("\nBaseline methods tests completed!")
