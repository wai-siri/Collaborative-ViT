"""
全局配置文件
统一管理模型路径、缓存目录等配置
"""

import os
import glob

# 项目根目录 (src 文件夹的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型缓存目录配置
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, 'models', 'pretrained')

# 如果模型缓存目录存在，设置环境变量
if os.path.exists(MODEL_CACHE_DIR):
    os.environ['TORCH_HOME'] = MODEL_CACHE_DIR
    os.environ['HF_HOME'] = MODEL_CACHE_DIR
    # 离线模式：禁止 HuggingFace Hub 联网（适用于无外网的云服务器）
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print(f"[Config] Using local model cache: {MODEL_CACHE_DIR}")
else:
    print(f"[Config] Model cache directory not found, will use default cache")
    print(f"[Config] Expected location: {MODEL_CACHE_DIR}")

# 查找本地预训练权重文件（兼容旧版 .npz 和新版 .safetensors/.bin）
LOCAL_CHECKPOINT_PATH = None
_search_dirs = [
    os.path.join(MODEL_CACHE_DIR, 'hub', 'checkpoints'),
    MODEL_CACHE_DIR,
]
_supported_exts = ['*.npz', '*.safetensors', '*.bin', '*.pth']
for _d in _search_dirs:
    if os.path.isdir(_d):
        for _ext in _supported_exts:
            _files = glob.glob(os.path.join(_d, _ext))
            if _files:
                LOCAL_CHECKPOINT_PATH = _files[0]
                break
    if LOCAL_CHECKPOINT_PATH:
        break

if LOCAL_CHECKPOINT_PATH:
    print(f"[Config] Found local checkpoint: {LOCAL_CHECKPOINT_PATH}")
else:
    print(f"[Config] No local checkpoint found, will try online download")

# 模型配置
MODEL_NAME = 'vit_large_patch16_384'
PRETRAINED = True

# 实验配置
DEFAULT_SLA = 300.0  # ms
DEFAULT_BATCH_SIZE = 1
PRUNE_PER_LAYER = 23  # 每层固定裁剪 token 数（论文 Fig.7 baseline = 23）
SPLIT_K = 5           # fine-to-coarse 候选点密度参数（论文参数 k）

# 网络配置
DEFAULT_SERVER_IP = '127.0.0.1'
DEFAULT_SERVER_PORT = 9999

# 数据路径
IMAGENET_DATA_PATH = 'data/validation-00000-of-00014.parquet'
NETWORK_TRACE_DIR = 'assets/network_traces'
PROFILER_DIR = 'assets/profiler_k_b'

# 结果保存路径
RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
