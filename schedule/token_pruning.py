import math
import torch


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


def prune_tokens(x, keep_n): # delete patch tokens
    B, T, D = x.shape # B: 批次大小, T: token 数, D: 特征维度

    if T <= keep_n:
        return x

    keep_n = max(keep_n, 1)

    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]

    scores = torch.norm(patch_tokens, p=2, dim=-1) # shape: (B, T-1)

    # keep cls_token and keep_n - 1 patch tokens
    num_keep_patch = keep_n - 1
    if num_keep_patch <= 0:
        return cls_token

    # _: 值, top_indices: 索引
    _, top_indices = torch.topk(scores, k=num_keep_patch, dim=-1, largest=True, sorted=True)
    top_indices, _ = torch.sort(top_indices, dim=-1)

    expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, D)
    selected_patches = torch.gather(patch_tokens, dim=1, index=expanded_indices)

    x_pruned = torch.cat([cls_token, selected_patches], dim=1)  # (B, keep_n, D)

    return x_pruned
