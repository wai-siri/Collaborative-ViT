#!/usr/bin/env python3
"""
Janus 云服务器环境检测脚本
==========================
在云服务器终端运行此脚本，自动检测运行 Janus 项目所需的全部环境依赖。

检测项目：
  1. Python 版本（要求 3.8+）
  2. PyTorch 及 CUDA/GPU 可用性
  3. 所有必需 Python 包
  4. 项目目录结构与关键文件
  5. ToMe 本地仓库完整性
  6. Profiler 数据文件
  7. 网络 trace 数据文件
  8. ImageNet 数据集
  9. 预训练模型缓存
 10. 磁盘空间与内存概览

用法：
  cd Janus/
  python src/check_env.py
"""

import os
import sys
import platform
import shutil

# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

# 颜色输出（终端支持 ANSI 时生效）
def _supports_color():
    """判断终端是否支持 ANSI 颜色"""
    if os.name == 'nt':
        return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

USE_COLOR = _supports_color()

def green(text):  return f"\033[92m{text}\033[0m" if USE_COLOR else text
def red(text):    return f"\033[91m{text}\033[0m" if USE_COLOR else text
def yellow(text): return f"\033[93m{text}\033[0m" if USE_COLOR else text
def bold(text):   return f"\033[1m{text}\033[0m" if USE_COLOR else text

PASS = green("[PASS]")
FAIL = red("[FAIL]")
WARN = yellow("[WARN]")
INFO = "[INFO]"

# 统计
total_checks = 0
passed_checks = 0
failed_checks = 0
warnings = 0

def check_pass(msg):
    global total_checks, passed_checks
    total_checks += 1
    passed_checks += 1
    print(f"  {PASS} {msg}")

def check_fail(msg, hint=""):
    global total_checks, failed_checks
    total_checks += 1
    failed_checks += 1
    print(f"  {FAIL} {msg}")
    if hint:
        print(f"        -> 修复建议: {hint}")

def check_warn(msg, hint=""):
    global warnings
    warnings += 1
    print(f"  {WARN} {msg}")
    if hint:
        print(f"        -> 建议: {hint}")

def check_info(msg):
    print(f"  {INFO} {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {bold(title)}")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
#  确定项目根目录
# ════════════════════════════════════════════════════════════════

# 本脚本位于 src/ 下，项目根目录是上一级
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = SCRIPT_DIR  # src/

print(f"\n{'#'*60}")
print(f"  Janus 云服务器环境检测")
print(f"{'#'*60}")
print(f"  项目根目录: {PROJECT_ROOT}")
print(f"  脚本位置:   {os.path.abspath(__file__)}")
print(f"  运行时间:   {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ════════════════════════════════════════════════════════════════
#  1. Python 版本
# ════════════════════════════════════════════════════════════════
section("1. Python 环境")

py_version = sys.version_info
check_info(f"Python 版本: {platform.python_version()}  ({sys.executable})")
if py_version >= (3, 8):
    check_pass(f"Python >= 3.8")
else:
    check_fail(f"Python {platform.python_version()} 不满足要求 (需要 3.8+)",
               "安装 Python 3.8 及以上版本")

check_info(f"操作系统: {platform.system()} {platform.release()} ({platform.machine()})")


# ════════════════════════════════════════════════════════════════
#  2. PyTorch 与 CUDA
# ════════════════════════════════════════════════════════════════
section("2. PyTorch 与 GPU")

torch_ok = False
try:
    import torch
    torch_ok = True
    check_pass(f"torch {torch.__version__}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            check_pass(f"GPU {i}: {name}  ({mem:.1f} GB)")
        check_info(f"CUDA 版本: {torch.version.cuda}")
        # 测试 GPU 是否真正可用
        try:
            t = torch.randn(2, 2, device='cuda')
            del t
            check_pass("CUDA 张量创建测试通过")
        except Exception as e:
            check_fail(f"CUDA 张量创建失败: {e}",
                       "检查 CUDA 驱动版本是否与 PyTorch 匹配")
    else:
        check_warn("CUDA 不可用，将使用 CPU（推理速度会很慢）",
                    "pip install torch --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    check_fail("torch 未安装",
               "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


# ════════════════════════════════════════════════════════════════
#  3. torchvision
# ════════════════════════════════════════════════════════════════
section("3. torchvision")

try:
    import torchvision
    check_pass(f"torchvision {torchvision.__version__}")
except ImportError:
    check_fail("torchvision 未安装",
               "pip install torchvision")


# ════════════════════════════════════════════════════════════════
#  4. 必需 Python 包
# ════════════════════════════════════════════════════════════════
section("4. 必需 Python 包")

# (包名, import名, 安装命令, 是否必须)
REQUIRED_PACKAGES = [
    ("timm",        "timm",      "pip install timm",              True),
    ("pandas",      "pandas",    "pip install pandas",            True),
    ("numpy",       "numpy",     "pip install numpy",             True),
    ("tqdm",        "tqdm",      "pip install tqdm",              True),
    ("pyarrow",     "pyarrow",   "pip install pyarrow",           True),
    ("PIL/Pillow",  "PIL",       "pip install Pillow",            True),
    ("matplotlib",  "matplotlib","pip install matplotlib",        False),
    ("scipy",       "scipy",     "pip install scipy",             False),
]

missing_required = []
for pkg_name, import_name, install_cmd, required in REQUIRED_PACKAGES:
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        check_pass(f"{pkg_name} {version}")
    except ImportError:
        if required:
            check_fail(f"{pkg_name} 未安装（必需）", install_cmd)
            missing_required.append(install_cmd)
        else:
            check_warn(f"{pkg_name} 未安装（可选，用于可视化等）", install_cmd)

# 特别检查 timm 版本是否支持 ViT-L
try:
    import timm
    if hasattr(timm, 'list_models'):
        vit_models = timm.list_models('vit_large*')
        if 'vit_large_patch16_384' in vit_models:
            check_pass("timm 包含 vit_large_patch16_384 模型")
        else:
            check_fail("timm 中找不到 vit_large_patch16_384",
                       "pip install timm --upgrade")
except Exception:
    pass


# ════════════════════════════════════════════════════════════════
#  5. 项目目录结构
# ════════════════════════════════════════════════════════════════
section("5. 项目目录结构")

REQUIRED_DIRS = [
    "src",
    "src/schedule",
    "src/profiler",
    "src/utils",
    "src/ToMe",
    "src/ToMe/ToMe",
    "assets",
    "assets/profiler_k_b",
    "assets/network_traces",
    "data",
    "results",
]

for d in REQUIRED_DIRS:
    full = os.path.join(PROJECT_ROOT, d)
    if os.path.isdir(full):
        check_pass(f"目录存在: {d}/")
    else:
        check_fail(f"目录缺失: {d}/",
                   f"mkdir -p {d}")


# ════════════════════════════════════════════════════════════════
#  6. 关键源代码文件
# ════════════════════════════════════════════════════════════════
section("6. 关键源代码文件")

REQUIRED_FILES = [
    "src/config.py",
    "src/download_model.py",
    "src/schedule/schedule.py",
    "src/schedule/declining_rate.py",
    "src/schedule/token_pruning.py",
    "src/schedule/split_inference.py",
    "src/profiler/test_time.py",
    "src/profiler/simulated.py",
    "src/utils/imagenet_loader.py",
    "src/utils/parse_network_traces.py",
    "src/simulation/device_only.py",
]

for f in REQUIRED_FILES:
    full = os.path.join(PROJECT_ROOT, f)
    if os.path.isfile(full):
        check_pass(f"文件存在: {f}")
    else:
        check_fail(f"文件缺失: {f}",
                   "请确认代码是否完整上传")


# ════════════════════════════════════════════════════════════════
#  7. ToMe 本地仓库
# ════════════════════════════════════════════════════════════════
section("7. ToMe（Token Merging）本地仓库")

tome_dir = os.path.join(SRC_DIR, "ToMe")
tome_pkg = os.path.join(tome_dir, "ToMe")
tome_merge = os.path.join(tome_pkg, "merge.py")

if os.path.isdir(tome_dir):
    check_pass("src/ToMe/ 目录存在")
else:
    check_fail("src/ToMe/ 目录缺失",
               "git clone https://github.com/facebookresearch/ToMe.git src/ToMe")

if os.path.isdir(tome_pkg):
    check_pass("src/ToMe/ToMe/ 包目录存在")
else:
    check_fail("src/ToMe/ToMe/ 包目录缺失")

if os.path.isfile(tome_merge):
    check_pass("src/ToMe/ToMe/merge.py 存在")
    # 尝试实际导入（仅导入 merge 模块，不经过 __init__.py）
    # 注意: ToMe 内部 patch/mae.py 使用 `from tome.utils import ...`（小写 tome），
    #       Linux 文件系统大小写敏感，整包导入会失败。
    #       项目只用到 merge.py 中的两个函数，因此直接导入 merge 模块即可。
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location("ToMe_merge", tome_merge)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        # 验证需要的函数存在
        assert hasattr(_mod, 'bipartite_soft_matching'), "缺少 bipartite_soft_matching"
        assert hasattr(_mod, 'merge_wavg'), "缺少 merge_wavg"
        check_pass("ToMe.merge 模块可正常导入 (bipartite_soft_matching, merge_wavg)")
    except Exception as e:
        check_fail(f"ToMe.merge 导入失败: {e}",
                   "检查 src/ToMe/ToMe/merge.py 文件完整性")
else:
    check_fail("src/ToMe/ToMe/merge.py 缺失")


# ════════════════════════════════════════════════════════════════
#  8. Profiler 数据文件
# ════════════════════════════════════════════════════════════════
section("8. Profiler 数据文件")

profiler_dir = os.path.join(PROJECT_ROOT, "assets", "profiler_k_b")
profiler_files = {
    "device_k_b.json": "设备端 profiler 参数",
    "cloud_k_b.json":  "云端 profiler 参数",
}

for fname, desc in profiler_files.items():
    fpath = os.path.join(profiler_dir, fname)
    if os.path.isfile(fpath):
        # 验证 JSON 格式
        try:
            import json
            with open(fpath, 'r') as f:
                data = json.load(f)
            num_layers = len([k for k in data.keys() if k.isdigit()])
            check_pass(f"{fname} ({desc}, {num_layers} 层)")
        except Exception as e:
            check_fail(f"{fname} 格式错误: {e}",
                       f"python src/profiler/simulated.py  # 重新生成")
    else:
        check_fail(f"{fname} 缺失 ({desc})",
                   f"python src/profiler/simulated.py  # 生成模拟数据\n"
                   f"        或 python src/profiler/test_time.py {fname}  # 实测生成")


# ════════════════════════════════════════════════════════════════
#  9. 网络 Trace 数据
# ════════════════════════════════════════════════════════════════
section("9. 网络 Trace 数据")

trace_dir = os.path.join(PROJECT_ROOT, "assets", "network_traces")
EXPECTED_TRACES = [
    "static_lte_trace.csv",
    "walking_lte_trace.csv",
    "driving_lte_trace.csv",
    "static_5g_trace.csv",
    "walking_5g_trace.csv",
    "driving_5g_trace.csv",
]

for fname in EXPECTED_TRACES:
    fpath = os.path.join(trace_dir, fname)
    if os.path.isfile(fpath):
        try:
            import pandas as pd
            df = pd.read_csv(fpath)
            rows = len(df)
            if 'bandwidth_mbps' in df.columns:
                avg_bw = df['bandwidth_mbps'].mean()
                check_pass(f"{fname}  ({rows} 行, 平均 {avg_bw:.2f} Mbps)")
            else:
                check_warn(f"{fname} 缺少 bandwidth_mbps 列")
        except Exception as e:
            check_warn(f"{fname} 读取异常: {e}")
    else:
        check_fail(f"{fname} 缺失",
                   "python src/utils/parse_network_traces.py  # 从原始数据生成")


# ════════════════════════════════════════════════════════════════
# 10. ImageNet 数据集
# ════════════════════════════════════════════════════════════════
section("10. ImageNet 数据集")

data_dir = os.path.join(PROJECT_ROOT, "data")
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')] if os.path.isdir(data_dir) else []

if parquet_files:
    for pf in parquet_files:
        fpath = os.path.join(data_dir, pf)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        check_pass(f"data/{pf}  ({size_mb:.1f} MB)")
    # 尝试读取一行验证格式
    try:
        import pandas as pd
        df = pd.read_parquet(os.path.join(data_dir, parquet_files[0]), engine='pyarrow')
        check_pass(f"Parquet 文件可读取, 共 {len(df)} 个样本")
        if 'image' in df.columns and 'label' in df.columns:
            check_pass("数据包含 'image' 和 'label' 列")
        else:
            check_warn(f"数据列: {list(df.columns)}, 预期包含 'image' 和 'label'")
    except Exception as e:
        check_warn(f"Parquet 文件读取测试失败: {e}")
else:
    check_warn("data/ 目录中未找到 .parquet 文件",
               "请将 ImageNet 验证集 parquet 文件放入 data/ 目录")


# ════════════════════════════════════════════════════════════════
# 11. 预训练模型缓存
# ════════════════════════════════════════════════════════════════
section("11. 预训练模型缓存")

model_cache = os.path.join(PROJECT_ROOT, "models", "pretrained")
if os.path.isdir(model_cache):
    # 计算缓存目录大小
    cache_size = 0
    file_count = 0
    for root, dirs, files in os.walk(model_cache):
        for f in files:
            cache_size += os.path.getsize(os.path.join(root, f))
            file_count += 1
    cache_size_mb = cache_size / (1024 * 1024)
    check_info(f"模型缓存目录: {model_cache}")
    check_info(f"缓存大小: {cache_size_mb:.1f} MB ({file_count} 个文件)")

    if cache_size_mb > 500:
        check_pass("模型缓存看起来已就绪 (>500MB)")
    else:
        check_warn("模型缓存较小，可能需要重新下载",
                   "python src/download_model.py")
else:
    check_warn("模型缓存目录不存在，首次运行时将自动下载 (~1.2GB)",
               "python src/download_model.py  # 提前手动下载")


# ════════════════════════════════════════════════════════════════
# 12. 模块导入测试
# ════════════════════════════════════════════════════════════════
section("12. 项目模块导入测试")

# 临时加入 src/ 到路径
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 测试各模块是否可正常导入
MODULE_TESTS = [
    ("config",                     "全局配置"),
    ("schedule.declining_rate",    "衰减率计算"),
    ("schedule.schedule",          "调度器"),
    ("schedule.token_pruning",     "Token 裁剪"),
    ("schedule.split_inference",   "模型分割推理"),
    ("utils.imagenet_loader",      "ImageNet 加载器"),
]

for mod_name, desc in MODULE_TESTS:
    try:
        __import__(mod_name)
        check_pass(f"import {mod_name}  ({desc})")
    except Exception as e:
        check_fail(f"import {mod_name} 失败: {e}")


# ════════════════════════════════════════════════════════════════
# 13. 磁盘空间与内存
# ════════════════════════════════════════════════════════════════
section("13. 系统资源")

# 磁盘空间
try:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    check_info(f"磁盘空间: {free_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计")
    if free_gb < 5:
        check_warn("磁盘可用空间不足 5GB，可能不够下载模型和存储结果",
                   "清理磁盘空间或扩容")
    else:
        check_pass(f"磁盘空间充足 ({free_gb:.1f} GB 可用)")
except Exception:
    check_info("无法获取磁盘空间信息")

# 内存
try:
    import psutil
    mem = psutil.virtual_memory()
    check_info(f"内存: {mem.available/(1024**3):.1f} GB 可用 / {mem.total/(1024**3):.1f} GB 总计")
    if mem.available / (1024**3) < 4:
        check_warn("可用内存不足 4GB，加载模型可能会出问题")
except ImportError:
    # 不安装 psutil 也没关系，尝试读 /proc/meminfo（Linux）
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                meminfo[key] = int(val)  # kB
        total_gb = meminfo.get('MemTotal', 0) / (1024**2)
        avail_gb = meminfo.get('MemAvailable', 0) / (1024**2)
        check_info(f"内存: {avail_gb:.1f} GB 可用 / {total_gb:.1f} GB 总计")
        if avail_gb < 4:
            check_warn("可用内存不足 4GB，加载模型可能会出问题")
    except Exception:
        check_info("无法获取内存信息（可安装 psutil: pip install psutil）")

# GPU 显存
if torch_ok and torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        mem_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024**3)
        check_info(f"GPU {i} 显存: {mem_free:.1f} GB 可用 / {mem_total:.1f} GB 总计")
        if mem_total < 4:
            check_warn(f"GPU {i} 显存小于 4GB，运行 ViT-L 可能不足",
                       "考虑使用更大显存的 GPU 或使用 CPU 模式")


# ════════════════════════════════════════════════════════════════
#  汇总
# ════════════════════════════════════════════════════════════════
print(f"\n{'#'*60}")
print(f"  检测完成")
print(f"{'#'*60}")
print(f"  通过: {green(str(passed_checks))}")
print(f"  失败: {red(str(failed_checks)) if failed_checks > 0 else str(failed_checks)}")
print(f"  警告: {yellow(str(warnings)) if warnings > 0 else str(warnings)}")
print(f"  总计: {total_checks}")

if failed_checks == 0:
    print(f"\n  {green('✓ 所有必需检测项均已通过！环境准备就绪。')}")
    print(f"  可以开始运行实验:")
    print(f"    python src/simulation/device_only.py")
else:
    print(f"\n  {red('✗ 存在未通过的检测项，请根据上方修复建议逐一解决。')}")
    if missing_required:
        print(f"\n  一键安装缺失的必需包:")
        print(f"    {' && '.join(missing_required)}")
        print(f"\n  或一次性安装全部依赖:")
        print(f"    pip install torch torchvision timm pandas tqdm pyarrow Pillow matplotlib numpy")

if warnings > 0:
    print(f"\n  {yellow('! 存在警告项，建议关注但不影响基本运行。')}")

print()
