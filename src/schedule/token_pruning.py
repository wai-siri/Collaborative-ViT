import math
import os
import sys
import torch

# 将本地克隆的 ToMe 仓库根目录加入路径（src/ToMe/），使内部 ToMe 包可导入
_TOME_REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ToMe")
if _TOME_REPO_DIR not in sys.path:
    sys.path.insert(0, _TOME_REPO_DIR)

# 从 src/ToMe/ToMe/merge.py 导入（路径已指向 src/ToMe/，所以 import ToMe.merge 即可）
from ToMe.merge import bipartite_soft_matching, merge_wavg


def compute_token_schedule(alpha, N, x_0):
    x_l = {}
    remaining = x_0 # 初始 token 数

    if alpha == 0:
        for l in range(1, N + 1):
            x_l[l] = x_0
    else:
        for l in range(1, N + 1):
            delta = math.floor(2 ** (alpha * (N - (l - 1))))
            remaining = remaining - delta
            x_l[l] = max(remaining, 1)

    return x_l # {l: keep_n}


def prune_tokens(x, keep_n):
    """
    Token pruning using ToMe (Token Merging) official algorithm.
    
    Uses Facebook Research's ToMe implementation with bipartite soft matching.
    This aligns with the paper's approach.
    
    Args:
        x: Input tensor (B, T, D) where T includes CLS token
        keep_n: Target number of tokens to keep
    
    Returns:
        Pruned tensor (B, keep_n, D)
    """
    B, T, D = x.shape

    if T <= keep_n:
        return x

    keep_n = max(keep_n, 1)

    # Calculate how many tokens to merge (r in ToMe paper)
    r = T - keep_n
    
    if r <= 0:
        return x
    
    # Use ToMe's bipartite_soft_matching
    # The metric is the token embeddings themselves (used for similarity)
    metric = x
    
    # Get merge and unmerge functions from ToMe
    # class_token=True because ViT has CLS token at position 0
    merge_fn, _ = bipartite_soft_matching(
        metric=metric,
        r=r,
        class_token=True,
        distill_token=False
    )
    
    # Apply merge with weighted average
    merged, _ = merge_wavg(merge_fn, x)
    
    return merged
