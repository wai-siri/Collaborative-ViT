"""
带宽估计模块
使用调和平均估计当前带宽
"""

def estimate_bandwidth(throughput_history, cold_start_default=100e6):
    """
    使用调和平均估计带宽
    
    Args:
        throughput_history: list of float, 历史吞吐量(bps)
        cold_start_default: float, 冷启动默认带宽(bps)
    
    Returns:
        float: 估计带宽(bps)
    """
    if not throughput_history:
        return cold_start_default
    
    # 过滤掉无效值
    valid_history = [x for x in throughput_history if x > 0]
    
    if not valid_history:
        return cold_start_default
    
    # 调和平均
    n = len(valid_history)
    harmonic_mean = n / sum(1.0 / x for x in valid_history)
    
    return harmonic_mean


if __name__ == "__main__":
    # 测试带宽估计
    print("Testing bandwidth estimator...")
    
    # 测试1: 空历史
    bw = estimate_bandwidth([])
    print(f"Empty history: {bw / 1e6:.2f} Mbps (expected: 100.00 Mbps)")
    
    # 测试2: 稳定带宽
    history = [100e6, 100e6, 100e6, 100e6]
    bw = estimate_bandwidth(history)
    print(f"Stable bandwidth: {bw / 1e6:.2f} Mbps (expected: 100.00 Mbps)")
    
    # 测试3: 波动带宽
    history = [100e6, 80e6, 120e6, 90e6]
    bw = estimate_bandwidth(history)
    print(f"Fluctuating bandwidth: {bw / 1e6:.2f} Mbps (harmonic mean)")
    
    # 测试4: 包含无效值
    history = [100e6, 0, 120e6, -10]
    bw = estimate_bandwidth(history)
    print(f"With invalid values: {bw / 1e6:.2f} Mbps")
    
    print("\nBandwidth estimator tests completed!")
