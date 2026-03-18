"""
下载ViT-L@384预训练模型到项目目录
"""

import os
import timm

# 设置缓存目录为项目内的models文件夹 (src 文件夹的上一级)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_cache_dir = os.path.join(project_root, 'models', 'pretrained')

# 创建目录
os.makedirs(model_cache_dir, exist_ok=True)

print("="*60)
print("Downloading ViT-L@384 Pretrained Model")
print("="*60)
print(f"Cache directory: {model_cache_dir}")
print()

# 设置环境变量，让timm使用项目内的缓存目录
os.environ['TORCH_HOME'] = model_cache_dir
os.environ['HF_HOME'] = model_cache_dir

print("Downloading model (this may take several minutes)...")
print("Model size: ~1.2GB")
print()

try:
    model = timm.create_model('vit_large_patch16_384', pretrained=True)
    print()
    print("="*60)
    print("✓ Model downloaded successfully!")
    print("="*60)
    print(f"Model saved to: {model_cache_dir}")
    print()
    print("Model info:")
    print(f"  - Layers: {len(model.blocks)}")
    print(f"  - Tokens: {model.pos_embed.size(1)}")
    print(f"  - Dimension: {model.pos_embed.size(2)}")
    print()
    print("Next steps:")
    print("  1. Upload the 'models' folder to AutoDL")
    print("  2. Update schedule.py to use this cache directory")
    
except Exception as e:
    print()
    print("="*60)
    print("✗ Download failed!")
    print("="*60)
    print(f"Error: {e}")
    print()
    print("Possible solutions:")
    print("  1. Check your internet connection")
    print("  2. Try using a VPN if Hugging Face is blocked")
    print("  3. Use mirror: set HF_ENDPOINT=https://hf-mirror.com")
