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
    """
    Device-side forward pass.
    
    split_layer 语义约定：
      - split_layer = 0:     cloud-only，device 只做 embedding，不执行任何 transformer layer
      - 1 <= split_layer <= N: 真实 split，device 执行 layer 1..split_layer
      - split_layer = N+1:   device-only，device 执行全部 N 层 transformer
    
    Args:
        model: ViT model
        image: input image tensor (B, 3, 384, 384)
        alpha: declining rate for token pruning
        split_layer: split point (0 to N+1)
    
    Returns:
        x: intermediate tensor after device-side layers
    """
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)

    x_l = compute_token_schedule(alpha, N, x_0)

    x = _embed(model, image)

    # split_layer <= 0: cloud-only，device 不执行任何 transformer layer
    if split_layer <= 0:
        return x

    # 执行 layer 1..min(split_layer, N)，覆盖 split_layer=N+1 时执行全部 N 层
    for l in range(1, min(split_layer, N) + 1):
        block_idx = l - 1
        x = model.blocks[block_idx](x)

        target_tokens = x_l[l]
        x = prune_tokens(x, target_tokens)

    return x # device output


#  Cloud ：layer (split_layer+1)..N + norm + head
def cloud_forward(model, x_mid, split_layer, alpha=0.0):
    """
    Cloud-side forward pass.
    
    split_layer 语义约定：
      - split_layer = 0:     cloud-only，cloud 从 layer 1 开始执行全部 N 层
      - 1 <= split_layer <= N: 真实 split，cloud 执行 layer (split_layer+1)..N
      - split_layer = N+1:   device-only，cloud 不执行任何 transformer layer，只做 norm + head
    
    同一个样本的整条推理链中，device 和 cloud 使用同一个 alpha，
    保证 mixed pruning 贯穿整条 ViT 层序列。
    
    Args:
        model: ViT model
        x_mid: intermediate tensor from device
        split_layer: split point (0 to N+1)
        alpha: declining rate, 必须与 device_forward 使用同一个值
    
    Returns:
        logits: model output
    """
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)
    
    # 始终计算 token schedule（alpha=0 时 x_l[l]=x_0，不裁剪）
    x_l = compute_token_schedule(alpha, N, x_0)
    
    x = x_mid
    # split_layer >= N+1 时不执行任何 transformer layer
    # split_layer = 0 时从 layer 1 开始执行全部 N 层
    for l in range(split_layer + 1, N + 1):
        block_idx = l - 1
        x = model.blocks[block_idx](x)
        
        # 对 cloud 端执行的每一层也应用同一个 alpha 的 pruning schedule
        target_tokens = x_l[l]
        x = prune_tokens(x, target_tokens)

    # norm + classification head（CLS token）
    x = model.norm(x)
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
        logits = cloud_forward(model, x_mid, split_layer, alpha)

    print(f"[Cloud] logits shape: {tuple(logits.shape)}")

    return logits, alpha, split_layer, x_mid_shape
