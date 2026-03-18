"""
全局配置文件
统一管理模型路径、缓存目录等配置
"""

import os

# 项目根目录 (src 文件夹的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型缓存目录配置
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, 'models', 'pretrained')

# 如果模型缓存目录存在，设置环境变量
if os.path.exists(MODEL_CACHE_DIR):
    os.environ['TORCH_HOME'] = MODEL_CACHE_DIR
    os.environ['HF_HOME'] = MODEL_CACHE_DIR
    print(f"[Config] Using local model cache: {MODEL_CACHE_DIR}")
else:
    print(f"[Config] Model cache directory not found, will use default cache")
    print(f"[Config] Expected location: {MODEL_CACHE_DIR}")

# 模型配置
MODEL_NAME = 'vit_large_patch16_384'
PRETRAINED = True

# 实验配置
DEFAULT_SLA = 300.0  # ms
DEFAULT_BATCH_SIZE = 1

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
