"""
Test simulation mode to verify it works correctly
This is a quick test before running full experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from schedule.schedule import init
from runtime.simulation import SimulatedJanus, SimulatedBaselines

def test_simulation_mode():
    """
    Test simulation mode with random data
    """
    print("="*80)
    print("Testing Janus Simulation Mode")
    print("="*80)
    
    # Load model
    print("\n1. Loading ViT-L@384 model...")
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"   Model loaded on {device}")
    print(f"   Layers: {len(model.blocks)}")
    print(f"   Initial tokens: {model.pos_embed.size(1)}")
    
    # Create simulators
    print("\n2. Creating simulators...")
    janus_sim = SimulatedJanus(model, device)
    baselines_sim = SimulatedBaselines(model, device)
    print("   Simulators created")
    
    # Test parameters
    test_cases = [
        {"name": "Low bandwidth (10 Mbps)", "bw": 10e6, "sla": 300.0},
        {"name": "Medium bandwidth (50 Mbps)", "bw": 50e6, "sla": 300.0},
        {"name": "High bandwidth (100 Mbps)", "bw": 100e6, "sla": 300.0},
    ]
    
    # Test with random images
    num_test_images = 5
    print(f"\n3. Testing with {num_test_images} random images...")
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test Case: {test_case['name']}")
        print(f"Bandwidth: {test_case['bw']/1e6:.1f} Mbps, SLA: {test_case['sla']:.0f}ms")
        print(f"{'='*80}")
        
        results = {
            'janus': {'latencies': [], 'alphas': [], 'splits': []},
            'device_only': {'latencies': []},
            'cloud_only': {'latencies': []},
            'mixed': {'latencies': []}
        }
        
        for i in range(num_test_images):
            # Random image
            image = torch.randn(1, 3, 384, 384, device=device)
            
            # Test Janus
            logits, lat, alpha, split = janus_sim.simulate_inference(
                image, test_case['bw'], test_case['sla']
            )
            results['janus']['latencies'].append(lat)
            results['janus']['alphas'].append(alpha)
            results['janus']['splits'].append(split)
            
            # Verify logits shape
            assert logits.shape == (1, 1000), f"Wrong logits shape: {logits.shape}"
            
            # Test Device-Only
            logits, lat = baselines_sim.device_only(image)
            results['device_only']['latencies'].append(lat)
            assert logits.shape == (1, 1000), f"Wrong logits shape: {logits.shape}"
            
            # Test Cloud-Only
            logits, lat = baselines_sim.cloud_only(image, test_case['bw'])
            results['cloud_only']['latencies'].append(lat)
            assert logits.shape == (1, 1000), f"Wrong logits shape: {logits.shape}"
            
            # Test Mixed
            logits, lat = baselines_sim.mixed(image, test_case['bw'], test_case['sla'])
            results['mixed']['latencies'].append(lat)
            assert logits.shape == (1, 1000), f"Wrong logits shape: {logits.shape}"
        
        # Print results
        print(f"\nResults (averaged over {num_test_images} images):")
        print(f"{'Method':<15} {'Avg Latency (ms)':<20} {'Extra Info':<30}")
        print("-"*65)
        
        janus_avg_lat = sum(results['janus']['latencies']) / len(results['janus']['latencies'])
        janus_avg_alpha = sum(results['janus']['alphas']) / len(results['janus']['alphas'])
        janus_avg_split = sum(results['janus']['splits']) / len(results['janus']['splits'])
        print(f"{'Janus':<15} {janus_avg_lat:<20.2f} α={janus_avg_alpha:.4f}, split={janus_avg_split:.1f}")
        
        device_avg_lat = sum(results['device_only']['latencies']) / len(results['device_only']['latencies'])
        print(f"{'Device-Only':<15} {device_avg_lat:<20.2f}")
        
        cloud_avg_lat = sum(results['cloud_only']['latencies']) / len(results['cloud_only']['latencies'])
        print(f"{'Cloud-Only':<15} {cloud_avg_lat:<20.2f}")
        
        mixed_avg_lat = sum(results['mixed']['latencies']) / len(results['mixed']['latencies'])
        print(f"{'Mixed':<15} {mixed_avg_lat:<20.2f}")
        
        # Check if Janus meets SLA
        violations = sum(1 for lat in results['janus']['latencies'] if lat > test_case['sla'])
        print(f"\nJanus SLA violations: {violations}/{num_test_images}")
        
        # Verify Janus is better or equal
        if janus_avg_lat <= min(device_avg_lat, cloud_avg_lat, mixed_avg_lat) * 1.1:  # Allow 10% tolerance
            print("✓ Janus achieves competitive latency")
        else:
            print("⚠ Janus latency higher than expected")
    
    print("\n" + "="*80)
    print("✓ Simulation mode test completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Prepare ImageNet data and network traces")
    print("2. Run full Fig.7 experiments with simulation mode")
    print("3. Analyze results and compare with paper")

if __name__ == "__main__":
    test_simulation_mode()
