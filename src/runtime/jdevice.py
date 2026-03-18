"""
Jdevice执行引擎
边缘设备端完整推理流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from schedule.schedule import init, schedule
from schedule.split_inference import device_forward
from schedule.declining_rate import declining_rate
from schedule.bandwidth_estimator import estimate_bandwidth
from runtime.compression import compress_tensor
from runtime.communication import DeviceClient


def run_jdevice(image, model, server_ip, server_port, SLA, bandwidth_history=None):
    """
    边缘设备端完整推理流程
    
    Args:
        image: torch.Tensor, 输入图像 [B, C, H, W]
        model: ViT模型
        server_ip: str, 云端服务器IP
        server_port: int, 云端服务器端口
        SLA: float, 延迟要求(ms)
        bandwidth_history: list, 历史带宽记录(bps)
    
    Returns:
        logits: torch.Tensor, 推理结果
        latency: float, 端到端延迟(ms)
        alpha: float, 使用的剪枝率
        split_point: int, 使用的分割点
    """
    start_time = time.time()
    
    # 1. 估计带宽
    if bandwidth_history is None:
        bandwidth_history = []
    bandwidth = estimate_bandwidth(bandwidth_history)
    
    # 2. 调度决策
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    D_M = model.pos_embed.size(2)
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits
    
    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)
    
    alpha, split_point = schedule(N, x_0, D_M, bits, num_steps, step, bandwidth, SLA)
    
    # 3. 设备端推理
    with torch.no_grad():
        x_mid = device_forward(model, image, alpha, split_point)
    
    # 4. 压缩并发送
    compressed = compress_tensor(x_mid)
    
    client = DeviceClient(server_ip, server_port)
    client.connect()
    bw = client.send_data(compressed, alpha, split_point)
    bandwidth_history.append(bw)
    
    # 5. 接收结果
    result_data = client.receive_result()
    client.close()
    
    # 解压结果
    from runtime.compression import decompress_tensor
    logits = decompress_tensor(result_data, device=image.device)
    
    # 计算延迟
    latency = (time.time() - start_time) * 1000  # ms
    
    return logits, latency, alpha, split_point


if __name__ == "__main__":
    print("Testing Jdevice execution engine...")
    print("Note: This requires Jcloud server to be running on the specified IP:port")
    print("For local testing, start Jcloud server first: python runtime/jcloud.py")
    
    # 加载模型
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 虚拟图像
    image = torch.randn(1, 3, 384, 384, device=device)
    
    # 运行（使用本地回环）
    SERVER_IP = '127.0.0.1'
    SERVER_PORT = 9999
    SLA = 100.0
    
    try:
        logits, latency, alpha, split = run_jdevice(
            image, model, SERVER_IP, SERVER_PORT, SLA
        )
        
        print(f"\nResults:")
        print(f"  Latency: {latency:.2f}ms")
        print(f"  α: {alpha:.4f}")
        print(f"  Split point: {split}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Top-5 classes: {torch.topk(logits, 5, dim=-1).indices.tolist()}")
        
        print("\nJdevice execution engine test completed!")
        
    except ConnectionRefusedError:
        print("\nError: Could not connect to Jcloud server")
        print("Please start the server first: python runtime/jcloud.py")
    except Exception as e:
        print(f"\nError: {e}")
