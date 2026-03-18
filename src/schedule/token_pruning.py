import math
import torch

# 导入ToMe官方API
try:
    from tome.merge import bipartite_soft_matching, merge_wavg
    TOME_AVAILABLE = True
except ImportError:
    TOME_AVAILABLE = False
    print("Warning: ToMe library not found. Token merging will use fallback implementation.")


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
    
    if not TOME_AVAILABLE:
        # Fallback: simple averaging (should not happen if ToMe is installed)
        raise RuntimeError("ToMe library is required but not available. Please install: pip install git+https://github.com/facebookresearch/ToMe.git")
    
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
