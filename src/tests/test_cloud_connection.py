"""
测试本地到AutoDL云端的连接
使用前请修改AUTODL_IP为您的AutoDL服务器IP
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from schedule.schedule import init
from runtime.jdevice import run_jdevice

# ===== AutoDL配置（通过SSH隧道） =====
AUTODL_IP = "127.0.0.1"  # 通过SSH隧道连接本地
AUTODL_PORT = 9999  # SSH隧道映射的端口
# =====================

def main():
    print("="*60)
    print("Testing Connection to AutoDL Cloud Server")
    print("="*60)
    print(f"Server: {AUTODL_IP}:{AUTODL_PORT}")
    print()
    
    # 加载模型
    print("Loading model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # 创建测试图像
    print("\nCreating test image...")
    image = torch.randn(1, 3, 384, 384, device=device)
    SLA = 300.0
    
    # 测试连接
    print(f"\nConnecting to {AUTODL_IP}:{AUTODL_PORT}...")
    print("(This may take a few seconds...)\n")
    
    try:
        logits, latency, alpha, split = run_jdevice(
            image, model, AUTODL_IP, AUTODL_PORT, SLA
        )
        
        print("="*60)
        print("✓ CONNECTION SUCCESSFUL!")
        print("="*60)
        print(f"  Latency:      {latency:.2f}ms")
        print(f"  α:            {alpha:.4f}")
        print(f"  Split point:  {split}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Top-5 classes: {torch.topk(logits, 5, dim=-1).indices.tolist()}")
        print("="*60)
        print("\nYou can now run the full experiments!")
        print("Next step: python experiments/run_imagenet.py --server_ip", AUTODL_IP)
        
    except ConnectionRefusedError:
        print("="*60)
        print("✗ CONNECTION FAILED: Connection Refused")
        print("="*60)
        print("\nPossible reasons:")
        print("  1. Jcloud server is not running on AutoDL")
        print("  2. IP address is incorrect")
        print("  3. Port is blocked")
        print("\nPlease check:")
        print("  - Is Jcloud server running? (python runtime/jcloud.py)")
        print("  - Is the IP address correct?")
        print("  - Can you ping the AutoDL server?")
        
    except TimeoutError:
        print("="*60)
        print("✗ CONNECTION FAILED: Timeout")
        print("="*60)
        print("\nPossible reasons:")
        print("  1. Network issue")
        print("  2. AutoDL server is not accessible")
        print("  3. Firewall blocking the connection")
        
    except Exception as e:
        print("="*60)
        print(f"✗ CONNECTION FAILED: {type(e).__name__}")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
