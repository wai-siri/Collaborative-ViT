"""
Simulation-based Evaluation for Janus

This module implements trace-driven simulation evaluation, which is the main
evaluation method used in the paper (Fig. 7, Fig. 9).

Architecture:
1. Load profiler data (device_k_b.json, cloud_k_b.json) from real prototype
2. Load network trace data (bandwidth over time)
3. For each sample:
   - Scheduler determines α and split point based on current bandwidth
   - Predict device latency using profiler
   - Calculate transmission latency = data_size / bandwidth
   - Predict cloud latency using profiler
   - Record total latency, accuracy, etc.
4. Output evaluation metrics (latency violation ratio, accuracy, throughput)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from schedule.schedule import schedule
from schedule.declining_rate import declining_rate
from schedule.token_pruning import compute_token_schedule


def load_profiler_data(profiler_dir=None):
    """Load profiler data from JSON files."""
    # 如果未指定 profiler_dir，使用项目根目录下的 assets/profiler_k_b
    if profiler_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/experiments/ 向上两级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(script_dir))
        profiler_dir = os.path.join(project_root, 'assets', 'profiler_k_b')
    
    device_file = Path(profiler_dir) / 'device_k_b.json'
    cloud_file = Path(profiler_dir) / 'cloud_k_b.json'
    
    with open(device_file, 'r') as f:
        device_profiler = json.load(f)
    
    with open(cloud_file, 'r') as f:
        cloud_profiler = json.load(f)
    
    return device_profiler, cloud_profiler


def load_network_trace(trace_file):
    """Load network trace from CSV file."""
    trace_df = pd.read_csv(trace_file)
    return trace_df


def predict_latency(num_tokens, profiler_data, layer_idx):
    """
    Predict inference latency using profiler data.
    
    Args:
        num_tokens: Number of tokens
        profiler_data: Profiler data dict
        layer_idx: Layer index
    
    Returns:
        Predicted latency in ms
    """
    k = profiler_data['k']
    b = profiler_data['b']
    
    # Linear model: latency = k * num_tokens + b
    latency_ms = k * num_tokens + b
    return latency_ms


def simulate_inference(model, image, bandwidth_mbps, SLA, device_profiler, cloud_profiler):
    """
    Simulate one inference using profiler data and current bandwidth.
    
    Args:
        model: ViT model
        image: Input image tensor
        bandwidth_mbps: Current bandwidth in Mbps
        SLA: Latency requirement in ms
        device_profiler: Device profiler data
        cloud_profiler: Cloud profiler data
    
    Returns:
        dict with latency, split_point, alpha, logits
    """
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    D_M = model.pos_embed.size(2)
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits
    
    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)
    
    bandwidth_bps = bandwidth_mbps * 1_000_000
    
    # Use scheduler to determine α and split point
    alpha, split_point = schedule(N, x_0, D_M, bits, num_steps, step, bandwidth_bps, SLA)
    
    # Compute token schedule
    x_l = compute_token_schedule(alpha, N, x_0)
    
    # Predict device latency
    device_latency = 0.0
    for l in range(split_point + 1):
        num_tokens = x_l[l] if l in x_l else x_0
        device_latency += predict_latency(num_tokens, device_profiler, l)
    
    # Calculate transmission latency
    if split_point < N + 1:
        num_tokens_transmitted = x_l[split_point] if split_point in x_l else x_0
        data_size_bits = num_tokens_transmitted * D_M * bits
        transmission_latency = (data_size_bits / bandwidth_bps) * 1000  # ms
    else:
        transmission_latency = 0.0
    
    # Predict cloud latency
    cloud_latency = 0.0
    for l in range(split_point + 1, N + 1):
        num_tokens = x_l[l] if l in x_l else x_0
        cloud_latency += predict_latency(num_tokens, cloud_profiler, l)
    
    total_latency = device_latency + transmission_latency + cloud_latency
    
    # Actually run the model to get accuracy (for now, simplified)
    with torch.no_grad():
        logits = model(image)
    
    return {
        'latency': total_latency,
        'device_latency': device_latency,
        'transmission_latency': transmission_latency,
        'cloud_latency': cloud_latency,
        'split_point': split_point,
        'alpha': alpha,
        'logits': logits,
        'num_tokens': x_l[N] if N in x_l else x_0
    }


def run_simulation(trace_file, num_samples=100, model_name='vit_large_patch16_384', SLA=300):
    """
    Run simulation evaluation.
    
    Args:
        trace_file: Path to network trace CSV
        num_samples: Number of samples to evaluate
        model_name: Model name
        SLA: Latency requirement in ms
    
    Returns:
        Results dict
    """
    print("=" * 60)
    print("Janus Simulation Evaluation")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"SLA: {SLA} ms")
    print(f"Samples: {num_samples}")
    print(f"Trace: {trace_file}")
    print()
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    
    # Load profiler data
    print("Loading profiler data...")
    device_profiler, cloud_profiler = load_profiler_data()
    
    # Load network trace
    print("Loading network trace...")
    trace_df = load_network_trace(trace_file)
    
    # Generate random images
    print("Generating test images...")
    img_size = model.default_cfg['input_size'][1]
    images = torch.randn(num_samples, 3, img_size, img_size, device=device)
    
    # Run simulation
    print("\nRunning simulation...")
    results = []
    
    for i in tqdm(range(num_samples)):
        # Get current bandwidth from trace (cycle through trace if needed)
        trace_idx = i % len(trace_df)
        bandwidth_mbps = trace_df.iloc[trace_idx]['bandwidth_mbps']
        
        # Simulate inference
        result = simulate_inference(
            model, images[i:i+1], bandwidth_mbps, SLA,
            device_profiler, cloud_profiler
        )
        
        result['sample_idx'] = i
        result['bandwidth_mbps'] = bandwidth_mbps
        results.append(result)
    
    # Calculate metrics
    latencies = [r['latency'] for r in results]
    violations = sum(1 for lat in latencies if lat > SLA)
    violation_ratio = violations / len(latencies)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_throughput = 1000.0 / avg_latency  # FPS
    
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"Latency Violation Ratio: {violation_ratio*100:.2f}%")
    print(f"Average Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} FPS")
    print(f"Latency Range: [{min(latencies):.2f}, {max(latencies):.2f}] ms")
    print("=" * 60)
    
    return {
        'results': results,
        'metrics': {
            'violation_ratio': violation_ratio,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'avg_throughput': avg_throughput,
            'min_latency': min(latencies),
            'max_latency': max(latencies)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Janus Simulation Evaluation')
    parser.add_argument('--trace', type=str, default='static',
                        choices=['static', 'walking', 'driving'],
                        help='Network trace scenario')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--model', type=str, default='vit_large_patch16_384',
                        help='Model name')
    parser.add_argument('--sla', type=int, default=300,
                        help='Latency requirement in ms')
    
    args = parser.parse_args()
    
    # 从 src/experiments/ 向上两级到达项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    trace_file = os.path.join(project_root, 'assets', 'network_traces', f'{args.trace}_trace.csv')
    
    results = run_simulation(
        trace_file=trace_file,
        num_samples=args.num_samples,
        model_name=args.model,
        SLA=args.sla
    )
    
    # Save results
    output_file = f'simulation_results_{args.trace}.json'
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {
            'metrics': results['metrics'],
            'num_samples': args.num_samples,
            'trace': args.trace,
            'sla': args.sla
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
