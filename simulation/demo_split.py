import torch
import torch.nn.functional as F

from ..schedule.schedule import init
from ..schedule.split_inference import (
    run_split_inference,
    full_forward,
)
from ..schedule.token_pruning import compute_token_schedule


def main():
    model = init()
    device = next(model.parameters()).device
    N = len(model.blocks)
    x_0 = model.pos_embed.size(1)

    # dummy image
    img_size = model.patch_embed.img_size[0]
    in_chans = model.patch_embed.proj.in_channels
    image = torch.randn(1, in_chans, img_size, img_size, device=device)

    # full forward
    with torch.no_grad():
        logits_full = full_forward(model, image)
    print(f"  logits shape: {tuple(logits_full.shape)}")
    print(f"  top-5 classes: {torch.topk(logits_full, 5, dim=-1).indices.tolist()}")

    test_bandwidths_mbps = [100, 200, 300, 500]
    SLA = 60.0

    for bw_mbps in test_bandwidths_mbps:
        bw_bps = bw_mbps * 1_000_000

        logits_split, alpha, split_layer, mid_shape = run_split_inference(
            model, image, bw_bps, SLA
        )

        cos_sim = F.cosine_similarity(logits_full, logits_split, dim=-1).item()
        x_l = compute_token_schedule(alpha, N, x_0)

        print(f"\n  [BW={bw_mbps}Mbps] α={alpha:.4f}, split={split_layer}, cos_sim={cos_sim:.4f}")
        print(f"    Tokens: {x_0} → {x_l[split_layer]} → {x_l[N]}")


if __name__ == "__main__":
    main()
