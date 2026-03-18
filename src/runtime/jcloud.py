"""
Jcloud执行引擎
云端推理服务器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from schedule.schedule import init
from schedule.split_inference import cloud_forward
from runtime.compression import decompress_tensor, compress_tensor
from runtime.communication import CloudServer


def run_jcloud_server(port, model=None):
    """
    云端推理服务器
    
    Args:
        port: int, 监听端口
        model: ViT模型（如果为None则自动加载）
    """
    if model is None:
        print("Loading model...")
        model = init()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded on {device}")
    
    server = CloudServer(port)
    print(f"Cloud server listening on port {port}...")
    
    try:
        while True:
            # 接受连接
            conn, addr = server.accept_connection()
            print(f"\nConnection from {addr}")
            
            try:
                # 接收数据
                compressed_data, alpha, split_point = server.receive_data(conn)
                print(f"Received: {len(compressed_data)} bytes, α={alpha:.4f}, split={split_point}")
                
                # 解压
                device = next(model.parameters()).device
                x_mid = decompress_tensor(compressed_data, device=device)
                print(f"Decompressed tensor shape: {x_mid.shape}")
                
                # 云端推理
                with torch.no_grad():
                    logits = cloud_forward(model, x_mid, split_point)
                print(f"Inference completed, logits shape: {logits.shape}")
                
                # 压缩结果
                result_data = compress_tensor(logits)
                
                # 发送结果
                server.send_result(conn, result_data)
                print(f"Result sent: {len(result_data)} bytes")
                
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
            finally:
                conn.close()
                
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    finally:
        server.close()
        print("Server closed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Jcloud Server')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    args = parser.parse_args()
    
    run_jcloud_server(args.port)
