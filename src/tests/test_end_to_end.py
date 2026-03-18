"""
端到端测试脚本
测试Jdevice和Jcloud的完整通信流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import threading
import time

from schedule.schedule import init
from runtime.jdevice import run_jdevice
from runtime.jcloud import run_jcloud_server


def server_thread_func(port, model):
    """服务器线程"""
    print("\n[Server Thread] Starting Jcloud server...")
    
    from runtime.communication import CloudServer
    from runtime.compression import decompress_tensor, compress_tensor
    from schedule.split_inference import cloud_forward
    
    server = CloudServer(port)
    print(f"[Server Thread] Listening on port {port}")
    
    # 只处理一个请求用于测试
    conn, addr = server.accept_connection()
    print(f"[Server Thread] Connection from {addr}")
    
    try:
        compressed_data, alpha, split_point = server.receive_data(conn)
        print(f"[Server Thread] Received: {len(compressed_data)} bytes, α={alpha:.4f}, split={split_point}")
        
        device = next(model.parameters()).device
        x_mid = decompress_tensor(compressed_data, device=device)
        print(f"[Server Thread] Decompressed tensor shape: {x_mid.shape}")
        
        with torch.no_grad():
            logits = cloud_forward(model, x_mid, split_point)
        print(f"[Server Thread] Inference completed, logits shape: {logits.shape}")
        
        result_data = compress_tensor(logits)
        server.send_result(conn, result_data)
        print(f"[Server Thread] Result sent: {len(result_data)} bytes")
        
    except Exception as e:
        print(f"[Server Thread] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        server.close()
        print("[Server Thread] Server closed")


def main():
    print("="*60)
    print("End-to-End Test: Jdevice + Jcloud")
    print("="*60)
    
    # 加载模型
    print("\nLoading model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # 启动服务器线程
    port = 9999
    server_thread = threading.Thread(target=server_thread_func, args=(port, model))
    server_thread.daemon = True
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(1)
    
    # 客户端测试
    print("\n" + "="*60)
    print("[Client] Starting Jdevice test...")
    print("="*60)
    
    image = torch.randn(1, 3, 384, 384, device=device)
    SLA = 100.0
    
    try:
        logits, latency, alpha, split = run_jdevice(
            image, model, '127.0.0.1', port, SLA
        )
        
        print("\n" + "="*60)
        print("Test Results:")
        print("="*60)
        print(f"  Latency:      {latency:.2f}ms")
        print(f"  α:            {alpha:.4f}")
        print(f"  Split point:  {split}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Top-5 classes: {torch.topk(logits, 5, dim=-1).indices.tolist()}")
        print("="*60)
        print("✓ End-to-End Test PASSED!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ End-to-End Test FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 等待服务器线程结束
    server_thread.join(timeout=2)


if __name__ == "__main__":
    main()
