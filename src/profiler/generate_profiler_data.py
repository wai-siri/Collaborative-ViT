"""
Generate profiler data for device and cloud
Based on paper's measurements and realistic hardware assumptions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import time
from tqdm import tqdm
from schedule.schedule import init

def measure_layer_latency(model, device, num_tokens_list=[100, 200, 300, 400, 500, 577], num_runs=10):
    """
    Measure actual layer latency for different token counts
    
    Args:
        model: ViT model
        device: torch device
        num_tokens_list: list of token counts to test
        num_runs: number of runs for averaging
        
    Returns:
        dict: {layer_idx: {token_count: latency_ms}}
    """
    model.eval()
    N = len(model.blocks)
    D = model.pos_embed.size(2)
    
    results = {}
    
    print(f"Measuring latency for {N} layers...")
    print(f"Token counts: {num_tokens_list}")
    print(f"Runs per measurement: {num_runs}")
    
    for layer_idx in tqdm(range(N), desc="Layers"):
        results[layer_idx] = {}
        block = model.blocks[layer_idx]
        
        for num_tokens in num_tokens_list:
            latencies = []
            
            # Warmup
            x = torch.randn(1, num_tokens, D, device=device)
            with torch.no_grad():
                _ = block(x)
            
            # Measure
            for _ in range(num_runs):
                x = torch.randn(1, num_tokens, D, device=device)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                with torch.no_grad():
                    _ = block(x)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)
            
            # Average
            avg_latency = sum(latencies) / len(latencies)
            results[layer_idx][num_tokens] = avg_latency
    
    return results


def fit_linear_model(measurements):
    """
    Fit linear model T = k * tokens + b for each layer
    
    Args:
        measurements: dict from measure_layer_latency
        
    Returns:
        dict: {layer_idx: {'k': k, 'b': b}}
    """
    import numpy as np
    
    profiler_data = {}
    
    for layer_idx, token_latencies in measurements.items():
        tokens = np.array(list(token_latencies.keys()))
        latencies = np.array(list(token_latencies.values()))
        
        # Linear regression using numpy (y = kx + b)
        # Using least squares: k = cov(x,y) / var(x), b = mean(y) - k * mean(x)
        n = len(tokens)
        mean_tokens = np.mean(tokens)
        mean_latencies = np.mean(latencies)
        
        # Calculate slope (k)
        numerator = np.sum((tokens - mean_tokens) * (latencies - mean_latencies))
        denominator = np.sum((tokens - mean_tokens) ** 2)
        k = numerator / denominator if denominator != 0 else 0
        
        # Calculate intercept (b)
        b = mean_latencies - k * mean_tokens
        
        # Calculate R-squared
        y_pred = k * tokens + b
        ss_res = np.sum((latencies - y_pred) ** 2)
        ss_tot = np.sum((latencies - mean_latencies) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        profiler_data[str(layer_idx)] = {
            'k': float(k),
            'b': float(b),
            'r_squared': float(r_squared)
        }
    
    return profiler_data


def generate_simulated_cloud_profiler(device_profiler_data, speedup_factor=10.0):
    """
    Generate simulated cloud profiler data based on device measurements
    
    Args:
        device_profiler_data: device profiler data
        speedup_factor: how much faster cloud is (default 10x based on paper)
        
    Returns:
        dict: cloud profiler data
    """
    cloud_profiler_data = {}
    
    for layer_idx, data in device_profiler_data.items():
        cloud_profiler_data[layer_idx] = {
            'k': data['k'] / speedup_factor,
            'b': data['b'] / speedup_factor,
            'r_squared': data.get('r_squared', 0.99),
            'p_value': data.get('p_value', 0.0),
            'simulated': True,
            'speedup_factor': speedup_factor
        }
    
    return cloud_profiler_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate profiler data')
    parser.add_argument('--mode', type=str, default='measure', 
                       choices=['measure', 'simulate'],
                       help='measure: actual measurement, simulate: use reasonable defaults')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to measure on')
    parser.add_argument('--cloud_speedup', type=float, default=10.0,
                       help='Cloud speedup factor vs device')
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'profiler_k_b'),
                       help='Output directory')
    parser.add_argument('--device_only', action='store_true',
                       help='Only generate device profiler data (do not overwrite cloud data)')
    args = parser.parse_args()
    
    print("="*80)
    print("Profiler Data Generation")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Cloud speedup: {args.cloud_speedup}x")
    print()
    
    # Load model
    print("Loading model...")
    model = init()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")
    print()
    
    if args.mode == 'measure':
        # Actual measurement
        print("Measuring device latency...")
        measurements = measure_layer_latency(model, device)
        
        print("\nFitting linear models...")
        device_profiler_data = fit_linear_model(measurements)
        
        # Print statistics
        print("\nDevice Profiler Statistics:")
        k_values = [data['k'] for data in device_profiler_data.values()]
        b_values = [data['b'] for data in device_profiler_data.values()]
        r2_values = [data['r_squared'] for data in device_profiler_data.values()]
        print(f"  k (slope): mean={sum(k_values)/len(k_values):.6f}, "
              f"min={min(k_values):.6f}, max={max(k_values):.6f}")
        print(f"  b (intercept): mean={sum(b_values)/len(b_values):.6f}, "
              f"min={min(b_values):.6f}, max={max(b_values):.6f}")
        print(f"  R²: mean={sum(r2_values)/len(r2_values):.4f}, "
              f"min={min(r2_values):.4f}")
        
    else:
        # Simulate based on paper's reported values
        print("Generating simulated device profiler data...")
        # Paper reports ~78ms for ViT-L on Jetson Orin Nano
        # With 24 layers and 577 tokens, estimate k and b
        device_profiler_data = {}
        for layer_idx in range(24):
            # Reasonable values based on paper
            device_profiler_data[str(layer_idx)] = {
                'k': 0.0085,  # ~0.0085ms per token
                'b': 0.32,    # ~0.32ms base latency
                'r_squared': 0.95,
                'p_value': 0.0,
                'simulated': True
            }
        print("  Using simulated values: k≈0.0085, b≈0.32")
    
    # Save device profiler
    os.makedirs(args.output_dir, exist_ok=True)
    device_path = os.path.join(args.output_dir, 'device_k_b.json')
    
    with open(device_path, 'w') as f:
        json.dump(device_profiler_data, f, indent=4)
    print(f"\n✓ Device profiler saved to: {device_path}")
    
    # Generate and save cloud profiler (unless device_only mode)
    if not args.device_only:
        print(f"\nGenerating cloud profiler data ({args.cloud_speedup}x speedup)...")
        cloud_profiler_data = generate_simulated_cloud_profiler(
            device_profiler_data, args.cloud_speedup
        )
        
        cloud_path = os.path.join(args.output_dir, 'cloud_k_b.json')
        with open(cloud_path, 'w') as f:
            json.dump(cloud_profiler_data, f, indent=4)
        print(f"✓ Cloud profiler saved to: {cloud_path}")
    else:
        print(f"\n⚠ Skipping cloud profiler generation (--device_only mode)")
        print(f"  Existing cloud_k_b.json will not be modified")
    
    print("\n" + "="*80)
    print("Profiler data generation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
