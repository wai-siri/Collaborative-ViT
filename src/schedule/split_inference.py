import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from schedule.token_pruning import compute_token_schedule, prune_tokens
from schedule.declining_rate import declining_rate
from schedule.schedule import schedule

# input preprocessing
def _embed(model, image):
    x = model.patch_embed(image)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls_token, x], dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    return x


def device_forward(model, image, alpha, split_layer):
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)

    x_l = compute_token_schedule(alpha, N, x_0)

    x = _embed(model, image)

    if split_layer <= 0:
        return x

    for l in range(1, min(split_layer, N) + 1):
        block_idx = l - 1
        x = model.blocks[block_idx](x)

        target_tokens = x_l[l]
        x = prune_tokens(x, target_tokens)

    return x # device output


#  Cloud ：layer (split_layer+1)..N + norm + head
def cloud_forward(model, x_mid, split_layer, alpha=0.0):
    """
    Cloud-side forward pass
    
    Args:
        model: ViT model
        x_mid: intermediate tensor from device
        split_layer: split point (layers 0 to split_layer executed on device)
        alpha: pruning rate (default 0.0 means no pruning on cloud side)
    
    Returns:
        logits: model output
    """
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    
    # Compute token schedule if pruning is needed
    if alpha > 0:
        x_l = compute_token_schedule(alpha, N, x_0)
    
    x = x_mid
    for l in range(split_layer + 1, N + 1):
        block_idx = l - 1
        x = model.blocks[block_idx](x)
        
        # Apply pruning if needed (typically not used on cloud side)
        if alpha > 0:
            target_tokens = x_l[l]
            x = prune_tokens(x, target_tokens)

    x = model.norm(x)

    # Take CLS token
    logits = model.head(x[:, 0])

    return logits


# no pruning
def full_forward(model, image):
    x = _embed(model, image)

    for block in model.blocks:
        x = block(x)

    x = model.norm(x)
    logits = model.head(x[:, 0])
    return logits


def run_split_inference(model, image, bandwidth_bps, SLA):
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    D_M = model.pos_embed.size(2)
    dtype = next(model.parameters()).dtype
    bits = torch.finfo(dtype).bits

    a_max = declining_rate(x_0, N)
    step = 0.01
    num_steps = int(a_max / step)

    alpha, split_layer = schedule(N, x_0, D_M, bits, num_steps, step, bandwidth_bps, SLA)

    print(f"[Scheduler] α={alpha:.4f}, split_layer={split_layer}")

    with torch.no_grad():
        x_mid = device_forward(model, image, alpha, split_layer)

    x_mid_shape = tuple(x_mid.shape)
    print(f"[Device] 中间张量 shape: {x_mid_shape}")

    with torch.no_grad():
        logits = cloud_forward(model, x_mid, split_layer)

    print(f"[Cloud] logits shape: {tuple(logits.shape)}")

    return logits, alpha, split_layer, x_mid_shape
