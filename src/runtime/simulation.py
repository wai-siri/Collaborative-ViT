"""
Single-machine simulation mode for Janus experiments
Simulates device-cloud collaboration without actual network communication
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import math

from schedule.schedule import schedule, device_profiler, cloud_profiler
from schedule.split_inference import device_forward, cloud_forward, full_forward
from schedule.declining_rate import declining_rate
from schedule.token_pruning import compute_token_schedule


class SimulatedJanus:
    """
    Simulated Janus system for single-machine experiments
    Uses profiler to predict latency instead of actual execution
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.N = len(model.blocks)
        self.x_0 = model.pos_embed.size(1)
        self.D_M = model.pos_embed.size(2)
        dtype = next(model.parameters()).dtype
        self.bits = torch.finfo(dtype).bits
        
    def simulate_inference(self, image, bandwidth_bps, SLA):
        """
        Simulate Janus inference with profiler-based latency prediction
        
        Args:
            image: torch.Tensor, input image
            bandwidth_bps: float, bandwidth in bps
            SLA: float, latency requirement in ms
            
        Returns:
            logits: torch.Tensor, inference result
            latency: float, simulated latency in ms
            alpha: float, pruning rate used
            split_point: int, split point used
        """
        start_time = time.time()
        
        # 1. Scheduler decision
        a_max = declining_rate(self.x_0, self.N)
        step = 0.01
        num_steps = int(a_max / step)
        
        alpha, split_point = schedule(
            self.N, self.x_0, self.D_M, self.bits, 
            num_steps, step, bandwidth_bps, SLA
        )
        
        # 2. Actual inference (to get correct accuracy)
        with torch.no_grad():
            if split_point == 0:
                # Cloud-only: full inference
                logits = full_forward(self.model, image)
            elif split_point == self.N + 1:
                # Device-only: full inference with pruning
                x_mid = device_forward(self.model, image, alpha, self.N)
                logits = cloud_forward(self.model, x_mid, self.N)
            else:
                # Split inference
                x_mid = device_forward(self.model, image, alpha, split_point)
                logits = cloud_forward(self.model, x_mid, split_point)
        
        # 3. Simulate latency using profiler
        simulated_latency = self._calculate_simulated_latency(
            alpha, split_point, bandwidth_bps
        )
        
        return logits, simulated_latency, alpha, split_point
    
    def _calculate_simulated_latency(self, alpha, split_point, bandwidth_bps):
        """
        Calculate simulated latency using profiler predictions
        
        Args:
            alpha: float, pruning rate
            split_point: int, split point
            bandwidth_bps: float, bandwidth in bps
            
        Returns:
            float, total latency in ms
        """
        # Compute token schedule
        x_l = compute_token_schedule(alpha, self.N, self.x_0)
        
        device_latency = 0.0
        cloud_latency = 0.0
        comm_latency = 0.0
        
        if split_point == 0:
            # Cloud-only
            for l in range(1, self.N + 1):
                cloud_latency += cloud_profiler(x_l[l], l)
            # Communication: send full image
            comm_latency = (self.x_0 * self.D_M * self.bits) / bandwidth_bps * 1000
            
        elif split_point == self.N + 1:
            # Device-only
            for l in range(1, self.N + 1):
                device_latency += device_profiler(x_l[l], l)
                
        else:
            # Split inference
            for l in range(1, split_point + 1):
                device_latency += device_profiler(x_l[l], l)
            
            for l in range(split_point + 1, self.N + 1):
                cloud_latency += cloud_profiler(x_l[l], l)
            
            # Communication: send intermediate tensor
            comm_latency = (x_l[split_point] * self.D_M * self.bits) / bandwidth_bps * 1000
        
        total_latency = device_latency + cloud_latency + comm_latency
        
        return total_latency


class SimulatedBaselines:
    """
    Simulated baseline methods
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.N = len(model.blocks)
        self.x_0 = model.pos_embed.size(1)
        self.D_M = model.pos_embed.size(2)
        dtype = next(model.parameters()).dtype
        self.bits = torch.finfo(dtype).bits
    
    def device_only(self, image, alpha_fixed=0.23):
        """
        Device-Only baseline with simulation
        
        Args:
            image: torch.Tensor, input image
            alpha_fixed: float, fixed pruning rate (paper uses 23 tokens/layer for ViT-L)
            
        Returns:
            logits: torch.Tensor, inference result
            latency: float, simulated latency in ms
        """
        # Actual inference
        with torch.no_grad():
            x_mid = device_forward(self.model, image, alpha_fixed, self.N)
            logits = cloud_forward(self.model, x_mid, self.N)
        
        # Simulate latency
        x_l = compute_token_schedule(alpha_fixed, self.N, self.x_0)
        latency = 0.0
        for l in range(1, self.N + 1):
            latency += device_profiler(x_l[l], l)
        
        return logits, latency
    
    def cloud_only(self, image, bandwidth_bps):
        """
        Cloud-Only baseline with simulation
        
        Args:
            image: torch.Tensor, input image
            bandwidth_bps: float, bandwidth in bps
            
        Returns:
            logits: torch.Tensor, inference result
            latency: float, simulated latency in ms
        """
        # Actual inference (no pruning)
        with torch.no_grad():
            logits = full_forward(self.model, image)
        
        # Simulate latency
        x_l = compute_token_schedule(0.0, self.N, self.x_0)  # No pruning
        latency = 0.0
        for l in range(1, self.N + 1):
            latency += cloud_profiler(x_l[l], l)
        
        # Add communication latency (send full image)
        comm_latency = (self.x_0 * self.D_M * self.bits) / bandwidth_bps * 1000
        latency += comm_latency
        
        return logits, latency
    
    def mixed(self, image, bandwidth_bps, SLA):
        """
        Mixed baseline: choose between Device-Only and Cloud-Only
        
        Args:
            image: torch.Tensor, input image
            bandwidth_bps: float, bandwidth in bps
            SLA: float, latency requirement in ms
            
        Returns:
            logits: torch.Tensor, inference result
            latency: float, simulated latency in ms
        """
        # Predict latencies for both options
        device_lat = 0.0
        x_l_device = compute_token_schedule(0.23, self.N, self.x_0)
        for l in range(1, self.N + 1):
            device_lat += device_profiler(x_l_device[l], l)
        
        cloud_lat = 0.0
        x_l_cloud = compute_token_schedule(0.0, self.N, self.x_0)
        for l in range(1, self.N + 1):
            cloud_lat += cloud_profiler(x_l_cloud[l], l)
        comm_lat = (self.x_0 * self.D_M * self.bits) / bandwidth_bps * 1000
        cloud_lat += comm_lat
        
        # Choose the better option
        if device_lat < cloud_lat:
            return self.device_only(image)
        else:
            return self.cloud_only(image, bandwidth_bps)


if __name__ == "__main__":
    from schedule.schedule import init
    
    print("Testing simulation mode...")
    
    # Load model
    model = init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create simulators
    janus_sim = SimulatedJanus(model, device)
    baselines_sim = SimulatedBaselines(model, device)
    
    # Test image
    image = torch.randn(1, 3, 384, 384, device=device)
    
    # Test parameters
    bandwidth_bps = 10e6  # 10 Mbps
    SLA = 300.0  # 300ms
    
    print("\n1. Testing Simulated Janus...")
    logits, lat, alpha, split = janus_sim.simulate_inference(image, bandwidth_bps, SLA)
    print(f"   Latency: {lat:.2f}ms (simulated)")
    print(f"   α: {alpha:.4f}, Split: {split}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Top-5 classes: {torch.topk(logits, 5, dim=-1).indices.tolist()}")
    
    print("\n2. Testing Device-Only...")
    logits, lat = baselines_sim.device_only(image)
    print(f"   Latency: {lat:.2f}ms (simulated)")
    print(f"   Logits shape: {logits.shape}")
    
    print("\n3. Testing Cloud-Only...")
    logits, lat = baselines_sim.cloud_only(image, bandwidth_bps)
    print(f"   Latency: {lat:.2f}ms (simulated)")
    print(f"   Logits shape: {logits.shape}")
    
    print("\n4. Testing Mixed...")
    logits, lat = baselines_sim.mixed(image, bandwidth_bps, SLA)
    print(f"   Latency: {lat:.2f}ms (simulated)")
    print(f"   Logits shape: {logits.shape}")
    
    print("\nSimulation mode tests completed!")
