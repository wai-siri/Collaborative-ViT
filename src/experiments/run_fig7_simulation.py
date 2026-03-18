"""
Fig.7 Simulation-based Experiments
Run all 6 scenarios using simulation mode (no dual-machine communication needed)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import json
import argparse
from tqdm import tqdm

from schedule.schedule import init
from runtime.simulation import SimulatedJanus, SimulatedBaselines
from utils.imagenet_loader import get_imagenet_loader


def load_trace(scenario, network_type, trace_dir=None):
    """Load network trace"""
    # 如果未指定 trace_dir，使用项目根目录下的 assets/network_traces
    if trace_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/experiments/ 向上两级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(base_dir))
        trace_dir = os.path.join(project_root, 'assets', 'network_traces')
    
    trace_path = os.path.join(trace_dir, f'{scenario}_{network_type}_trace.csv')
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace not found: {trace_path}")
    
    df = pd.read_csv(trace_path)
    print(f"Loaded trace: {scenario}_{network_type}")
    print(f"  Samples: {len(df)}")
    print(f"  Mean BW: {df['bandwidth_mbps'].mean():.2f} Mbps")
    print(f"  Std BW: {df['bandwidth_mbps'].std():.2f} Mbps")
    return df


def run_method_simulation(method_name, simulator, loader, trace_df, SLA):
    """
    Run a single method with simulation
    
    Args:
        method_name: str, method name
        simulator: SimulatedJanus or SimulatedBaselines
        loader: DataLoader
        trace_df: network trace DataFrame
        SLA: float, latency requirement in ms
    
    Returns:
        dict: results
    """
    import time
    
    latencies = []
    predictions = []
    labels = []
    alphas = []
    splits = []
    bandwidths = []
    
    trace_idx = 0
    
    for images, targets in tqdm(loader, desc=method_name):
        images = images.to(next(simulator.model.parameters()).device)
        
        # Get bandwidth from trace
        bw_mbps = trace_df.iloc[trace_idx % len(trace_df)]['bandwidth_mbps']
        bw_bps = bw_mbps * 1e6
        bandwidths.append(bw_mbps)
        trace_idx += 1
        
        try:
            # Measure actual inference time for all methods
            start_time = time.time()
            
            if method_name == 'janus':
                logits, _, alpha, split = simulator.simulate_inference(images, bw_bps, SLA)
                alphas.append(alpha)
                splits.append(split)
            elif method_name == 'device_only':
                logits, _ = simulator.device_only(images)
            elif method_name == 'cloud_only':
                logits, _ = simulator.cloud_only(images, bw_bps)
            elif method_name == 'mixed':
                logits, _ = simulator.mixed(images, bw_bps, SLA)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Use actual measured latency for fair comparison
            lat = (time.time() - start_time) * 1000
            
            latencies.append(lat)
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
            labels.append(targets.item())
            
        except Exception as e:
            print(f"\nError on sample {len(latencies)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute metrics
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(predictions) * 100 if predictions else 0
    
    violations = sum(1 for lat in latencies if lat > SLA)
    violation_ratio = violations / len(latencies) * 100 if latencies else 0
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = 1000 / avg_latency if avg_latency > 0 else 0
    
    results = {
        'method': method_name,
        'total_samples': len(latencies),
        'accuracy': accuracy,
        'avg_latency': avg_latency,
        'violation_ratio': violation_ratio,
        'throughput': throughput,
        'avg_bandwidth_mbps': sum(bandwidths) / len(bandwidths) if bandwidths else 0,
        'latencies': latencies,
        'predictions': predictions,
        'labels': labels
    }
    
    if alphas:
        results['avg_alpha'] = sum(alphas) / len(alphas)
        results['avg_split'] = sum(splits) / len(splits)
        results['alphas'] = alphas
        results['splits'] = splits
    
    return results


def run_scenario(scenario, network_type, model, loader, SLA, output_dir='results/fig7_simulation'):
    """
    Run all methods for one scenario
    """
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario.upper()} - {network_type.upper()}")
    print(f"{'='*80}")
    
    # Load trace
    trace_df = load_trace(scenario, network_type)
    
    # Create simulators
    device = next(model.parameters()).device
    janus_sim = SimulatedJanus(model, device)
    baselines_sim = SimulatedBaselines(model, device)
    
    # Run all methods
    methods = {
        'janus': janus_sim,
        'device_only': baselines_sim,
        'cloud_only': baselines_sim,
        'mixed': baselines_sim
    }
    
    all_results = {}
    
    for method_name, simulator in methods.items():
        print(f"\nRunning {method_name}...")
        results = run_method_simulation(method_name, simulator, loader, trace_df, SLA)
        all_results[method_name] = results
        
        # Print summary
        print(f"\n{method_name.upper()} Results:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Avg Latency: {results['avg_latency']:.2f} ms")
        print(f"  Violation Ratio: {results['violation_ratio']:.2f}%")
        print(f"  Throughput: {results['throughput']:.2f} FPS")
        if 'avg_alpha' in results:
            print(f"  Avg α: {results['avg_alpha']:.4f}")
            print(f"  Avg Split: {results['avg_split']:.1f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{scenario}_{network_type}_results.json')
    
    # Convert to serializable format
    save_results = {}
    for method, res in all_results.items():
        save_results[method] = {
            'method': res['method'],
            'total_samples': res['total_samples'],
            'accuracy': res['accuracy'],
            'avg_latency': res['avg_latency'],
            'violation_ratio': res['violation_ratio'],
            'throughput': res['throughput'],
            'avg_bandwidth_mbps': res['avg_bandwidth_mbps']
        }
        if 'avg_alpha' in res:
            save_results[method]['avg_alpha'] = res['avg_alpha']
            save_results[method]['avg_split'] = res['avg_split']
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run Fig.7 simulation experiments')
    parser.add_argument('--data_path', type=str, 
                       default='data/validation-00000-of-00014.parquet',
                       help='ImageNet parquet file path')
    parser.add_argument('--sla', type=float, default=300.0,
                       help='Latency SLA in ms')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       default=['static', 'walking', 'driving'],
                       help='Scenarios to test')
    parser.add_argument('--networks', type=str, nargs='+',
                       default=['lte', '5g'],
                       help='Network types to test')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples (None for all)')
    args = parser.parse_args()
    
    print("="*80)
    print("Fig.7 Simulation-based Experiments")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"SLA: {args.sla} ms")
    print(f"Scenarios: {args.scenarios}")
    print(f"Networks: {args.networks}")
    print("="*80)
    
    # Load model
    print("\nLoading ViT-L@384 model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Load data
    print(f"\nLoading ImageNet data...")
    
    if args.num_samples:
        # Create subset before DataLoader
        from utils.imagenet_loader import ImageNetParquetDataset
        from torchvision import transforms
        from torch.utils.data import DataLoader, Subset
        
        transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = ImageNetParquetDataset(args.data_path, transform=transform)
        indices = list(range(min(args.num_samples, len(full_dataset))))
        subset_dataset = Subset(full_dataset, indices)
        
        loader = DataLoader(
            subset_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"Limited to {len(subset_dataset)} samples for testing")
    else:
        loader = get_imagenet_loader(args.data_path, batch_size=1)
    
    print(f"Dataset size: {len(loader.dataset)}")
    
    # Run all scenarios
    all_scenario_results = {}
    
    for scenario in args.scenarios:
        for network in args.networks:
            key = f"{scenario}_{network}"
            try:
                results = run_scenario(scenario, network, model, loader, args.sla)
                all_scenario_results[key] = results
            except Exception as e:
                print(f"\n❌ Error in {key}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - All Scenarios")
    print("="*80)
    print(f"{'Scenario':<20} {'Method':<15} {'Acc(%)':<10} {'Lat(ms)':<12} {'Viol(%)':<10} {'FPS':<10}")
    print("-"*80)
    
    for scenario_key, results in all_scenario_results.items():
        for method, res in results.items():
            print(f"{scenario_key:<20} {method:<15} "
                  f"{res['accuracy']:<10.2f} {res['avg_latency']:<12.2f} "
                  f"{res['violation_ratio']:<10.2f} {res['throughput']:<10.2f}")
    
    print("="*80)
    print("\n✓ All experiments completed!")
    print(f"Results saved to: results/fig7_simulation/")


if __name__ == "__main__":
    main()
