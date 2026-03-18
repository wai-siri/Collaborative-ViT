"""
数据压缩模块
使用LZW压缩来压缩tensor数据，对齐论文实现
"""

import pickle
import torch


# ===== 标准LZW压缩算法实现 =====
def lzw_compress(data):
    """
    标准LZW压缩算法
    
    Args:
        data: bytes, 原始数据
    
    Returns:
        list of int, 压缩后的索引序列
    """
    # 初始化字典（0-255为单字节）
    dict_size = 256
    dictionary = {bytes([i]): i for i in range(dict_size)}
    
    result = []
    current = b""
    
    for byte in data:
        byte_val = bytes([byte])
        combined = current + byte_val
        
        if combined in dictionary:
            current = combined
        else:
            result.append(dictionary[current])
            # 限制字典大小避免内存溢出
            if dict_size < 65536:  # 使用16位索引
                dictionary[combined] = dict_size
                dict_size += 1
            current = byte_val
    
    if current:
        result.append(dictionary[current])
    
    return result


def lzw_decompress(compressed):
    """
    标准LZW解压算法
    
    Args:
        compressed: list of int, 压缩的索引序列
    
    Returns:
        bytes, 解压后的数据
    """
    if not compressed:
        return b""
    
    dict_size = 256
    dictionary = {i: bytes([i]) for i in range(dict_size)}
    
    result = []
    current = bytes([compressed[0]])
    result.append(current)
    
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = current + current[:1]
        else:
            raise ValueError(f"Invalid compressed code: {code}")
        
        result.append(entry)
        
        # 限制字典大小
        if dict_size < 65536:
            dictionary[dict_size] = current + entry[:1]
            dict_size += 1
        
        current = entry
    
    return b"".join(result)


def compress_tensor(tensor):
    """
    压缩PyTorch tensor使用LZW算法
    
    Args:
        tensor: torch.Tensor, 需要压缩的tensor
    
    Returns:
        bytes: 压缩后的数据
    """
    # 转为numpy并序列化
    tensor_bytes = pickle.dumps(tensor.cpu().numpy())
    
    # LZW压缩
    compressed_indices = lzw_compress(tensor_bytes)
    
    # 将索引列表转为bytes（使用pickle）
    compressed_bytes = pickle.dumps(compressed_indices)
    
    return compressed_bytes


def decompress_tensor(compressed_data, device='cpu'):
    """
    解压为PyTorch tensor
    
    Args:
        compressed_data: bytes, 压缩的数据
        device: str or torch.device, 目标设备
    
    Returns:
        torch.Tensor: 解压后的tensor
    """
    # 从bytes恢复索引列表
    compressed_indices = pickle.loads(compressed_data)
    
    # LZW解压
    decompressed_bytes = lzw_decompress(compressed_indices)
    
    # 反序列化
    numpy_array = pickle.loads(decompressed_bytes)
    
    # 转为tensor
    tensor = torch.from_numpy(numpy_array).to(device)
    
    return tensor


if __name__ == "__main__":
    print("Testing compression module...")
    
    # 测试不同大小的tensor
    test_cases = [
        (1, 577, 1024),   # ViT-L@384 初始token
        (1, 300, 1024),   # 中间剪枝后
        (1, 100, 1024),   # 大量剪枝后
    ]
    
    for shape in test_cases:
        x = torch.randn(*shape)
        
        # 压缩
        compressed = compress_tensor(x)
        
        # 解压
        x_recovered = decompress_tensor(compressed)
        
        # 计算压缩率和误差
        original_size = x.element_size() * x.nelement()
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        recovery_error = torch.abs(x - x_recovered).max().item()
        
        print(f"\nShape: {shape}")
        print(f"  Original size: {original_size / 1024:.2f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Recovery error: {recovery_error:.2e}")
    
    print("\nCompression module tests completed!")
