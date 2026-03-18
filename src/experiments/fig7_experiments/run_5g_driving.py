"""
Fig. 7实验 - 5G Driving场景
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import argparse

from utils.imagenet_loader import get_imagenet_loader
from experiments.fig7_experiments.common import load_trace, run_single_method, save_experiment_results, save_summary


def main():
    parser = argparse.ArgumentParser(description='Run Fig. 7 5G Driving experiment')
    parser.add_argument('--data_path', type=str, 
                       default='data/validation-00000-of-00014.parquet',
                       help='ImageNet parquet file path')
    parser.add_argument('--sla', type=float, default=300, 
                       help='Latency SLA in ms')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1',
                       help='Cloud server IP (for SSH tunnel)')
    parser.add_argument('--server_port', type=int, default=9999,
                       help='Cloud server port')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['device_only', 'janus', 'cloud_only', 'mixed'],
                       help='Methods to test')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    args = parser.parse_args()
    
    print("="*80)
    print("Fig. 7 Experiment - 5G Driving Scenario")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"SLA: {args.sla} ms")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Methods: {args.methods}")
    print("="*80 + "\n")
    
    # 加载模型
    print("Loading ViT model...")
    from schedule.schedule import init
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    
    # 加载ImageNet数据
    print("Loading ImageNet data...")
    loader = get_imagenet_loader(args.data_path, batch_size=args.batch_size)
    print(f"Loaded {len(loader.dataset)} images\n")
    
    # 加载网络trace
    print("Loading network trace...")
    trace_df = load_trace('driving', '5g')
    print()
    
    # 运行实验
    all_results = {}
    
    for method in args.methods:
        if method in ['janus', 'cloud_only', 'mixed']:
            print(f"\n⚠️  {method.upper()} requires cloud server connection!")
            print(f"   Please ensure Jcloud server is running on AutoDL")
            print(f"   and SSH tunnel is established: ssh -L {args.server_port}:localhost:9999 autodl")
            input("   Press Enter to continue...")
        
        try:
            result = run_single_method(
                method, model, loader, trace_df, args.sla,
                args.server_ip, args.server_port
            )
            all_results[method] = result
            
            output_path = f'results/fig7/5g_driving_{method}.json'
            save_experiment_results(result, output_path)
            
        except Exception as e:
            print(f"\n❌ Error running {method}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {method}...\n")
    
    if all_results:
        save_summary(all_results, 'results/fig7/5g_driving_summary.json')
    
    print("\n" + "="*80)
    print("✓ Experiment completed!")
    print("="*80)
    print("Results saved to:")
    print("  - results/fig7/5g_driving_*.json")
    print("  - results/fig7/5g_driving_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()
