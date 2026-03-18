"""
Generate synthetic network traces for experiments
Based on paper's reported network statistics
"""

import os
import pandas as pd
import numpy as np

def generate_trace(scenario, network_type, duration_seconds=3600, output_dir=None):
    """
    Generate network bandwidth trace
    
    Args:
        scenario: str, 'static', 'walking', or 'driving'
        network_type: str, '5g' or 'lte'
        duration_seconds: int, trace duration
        output_dir: str, output directory
    
    Returns:
        pd.DataFrame with columns: timestamp, bandwidth_mbps
    """
    # 如果未指定 output_dir，使用项目根目录下的 assets/network_traces
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 从 src/utils/ 向上两级到达项目根目录
        project_root = os.path.dirname(os.path.dirname(base_dir))
        output_dir = os.path.join(project_root, 'assets', 'network_traces')
    
    # Paper-reported statistics (Section V-B)
    # LTE: mean 7.6 Mbps
    # 5G: mean 14.7 Mbps
    
    if network_type == 'lte':
        base_bandwidth = 7.6  # Mbps
    elif network_type == '5g':
        base_bandwidth = 14.7  # Mbps
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    # Scenario-specific parameters
    if scenario == 'static':
        # Static: low variance, stable
        std_ratio = 0.2
        fluctuation_freq = 0.1  # Hz
    elif scenario == 'walking':
        # Walking: medium variance, moderate fluctuations
        std_ratio = 0.4
        fluctuation_freq = 0.3  # Hz
    elif scenario == 'driving':
        # Driving: high variance, frequent fluctuations (blockage, handover)
        std_ratio = 0.6
        fluctuation_freq = 0.5  # Hz
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Generate trace
    sample_rate = 10  # Hz (10 samples per second)
    num_samples = duration_seconds * sample_rate
    
    timestamps = np.arange(num_samples) / sample_rate
    
    # Base signal: mean + slow drift
    drift = np.sin(2 * np.pi * 0.01 * timestamps) * base_bandwidth * 0.1
    
    # Fast fluctuations
    fluctuations = np.sin(2 * np.pi * fluctuation_freq * timestamps) * base_bandwidth * std_ratio
    
    # Random noise
    noise = np.random.normal(0, base_bandwidth * std_ratio * 0.3, num_samples)
    
    # Combine
    bandwidth = base_bandwidth + drift + fluctuations + noise
    
    # Clip to reasonable range (min 1 Mbps)
    bandwidth = np.clip(bandwidth, 1.0, base_bandwidth * 3)
    
    # Add occasional deep fades for driving scenario
    if scenario == 'driving':
        num_fades = int(duration_seconds / 60)  # ~1 fade per minute
        fade_indices = np.random.choice(num_samples, num_fades, replace=False)
        for idx in fade_indices:
            fade_duration = np.random.randint(10, 50)  # 1-5 seconds
            fade_start = max(0, idx - fade_duration // 2)
            fade_end = min(num_samples, idx + fade_duration // 2)
            bandwidth[fade_start:fade_end] *= 0.2  # Drop to 20%
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'bandwidth_mbps': bandwidth
    })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{scenario}_{network_type}_trace.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Generated {filename}:")
    print(f"  Samples: {len(df)}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Mean BW: {df['bandwidth_mbps'].mean():.2f} Mbps")
    print(f"  Std BW: {df['bandwidth_mbps'].std():.2f} Mbps")
    print(f"  Min BW: {df['bandwidth_mbps'].min():.2f} Mbps")
    print(f"  Max BW: {df['bandwidth_mbps'].max():.2f} Mbps")
    print(f"  Saved to: {filepath}")
    print()
    
    return df


def main():
    """
    Generate all 6 network traces for Fig.7 experiments
    """
    print("="*80)
    print("Network Trace Generation")
    print("="*80)
    print()
    
    scenarios = ['static', 'walking', 'driving']
    network_types = ['lte', '5g']
    
    # For ImageNet experiments, we need enough samples
    # Assume ~2s per image, 3572 images = ~7200s = 2 hours
    duration = 7200
    
    for scenario in scenarios:
        for network_type in network_types:
            print(f"Generating {scenario}_{network_type} trace...")
            generate_trace(scenario, network_type, duration)
    
    print("="*80)
    print("✓ All network traces generated!")
    print("="*80)
    print()
    print("Generated traces:")
    for scenario in scenarios:
        for network_type in network_types:
            print(f"  - {scenario}_{network_type}_trace.csv")


if __name__ == "__main__":
    main()
