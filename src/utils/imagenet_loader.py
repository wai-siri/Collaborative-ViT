"""
ImageNet数据加载器
从parquet格式文件加载ImageNet验证集数据
"""

import io
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageNetParquetDataset(Dataset):
    """
    ImageNet Parquet格式数据集
    """
    
    def __init__(self, parquet_path, transform=None):
        """
        Args:
            parquet_path: str, parquet文件路径
            transform: torchvision.transforms, 图像变换
        """
        print(f"Loading ImageNet data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        print(f"Loaded {len(self.df)} images")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            tuple: (image_tensor, label, image_size_bytes)
                image_size_bytes: int, 原始图片编码后的字节数（JPEG/PNG），
                    用于 cloud-only 场景下计算真实图片传输通信时间
        """
        row = self.df.iloc[idx]
        
        # 解码图像字节流
        img_bytes = row['image']['bytes']
        image_size_bytes = len(img_bytes)  # 原始编码后的真实文件大小
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 获取标签
        label = row['label']
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        return img, label, image_size_bytes


def get_imagenet_loader(parquet_path, batch_size=1, num_workers=0, shuffle=False):
    """
    获取ImageNet DataLoader
    
    Args:
        parquet_path: str, parquet文件路径
        batch_size: int, batch大小
        num_workers: int, 数据加载线程数
        shuffle: bool, 是否打乱数据
    
    Returns:
        DataLoader
    """
    # ViT-L@384的标准预处理
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageNetParquetDataset(parquet_path, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader


def verify_imagenet_data(parquet_path):
    """
    验证ImageNet数据是否可以正确加载
    
    Args:
        parquet_path: str, parquet文件路径
    """
    print("="*60)
    print("Verifying ImageNet Data")
    print("="*60)
    
    # 加载数据
    loader = get_imagenet_loader(parquet_path, batch_size=1)
    
    print(f"\nDataset size: {len(loader.dataset)}")
    print(f"Batch size: {loader.batch_size}")
    
    # 测试加载第一个batch
    print("\nLoading first batch...")
    for images, labels, img_sizes in loader:
        print(f"Image shape: {images.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Label: {labels.item()}")
        print(f"Image file size: {img_sizes.item()} bytes ({img_sizes.item()/1024:.1f} KB)")
        break
    
    # 统计标签分布和图片大小
    print("\nLabel statistics:")
    all_labels = [label.item() for _, label, _ in loader]
    unique_labels = set(all_labels)
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Label range: [{min(all_labels)}, {max(all_labels)}]")
    
    print("\n" + "="*60)
    print("✓ ImageNet data verified successfully!")
    print("="*60)


if __name__ == "__main__":
    SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, SRC_DIR)
    import config

    parquet_path = os.path.join(config.PROJECT_ROOT, config.IMAGENET_DATA_PATH)

    if Path(parquet_path).exists():
        verify_imagenet_data(parquet_path)
    else:
        print(f"Error: File not found: {parquet_path}")
