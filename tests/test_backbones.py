"""Self-check for darkdiff.backbones with random (no-download) models. Run: uv run tests/test_backbones.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from diffusers import DDPMScheduler, UNet2DConditionModel  # noqa: E402

from darkdiff.model import DarkDiff  # noqa: E402
from test_model import DEV, tiny_rvae  # noqa: E402


def tiny_distilled_unet():
    # Same shape as bk-sdm-v2-tiny: no mid block, 3 down blocks. Channels shrunk.
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D",) * 2 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 2,
        block_out_channels=(32, 64, 64), layers_per_block=1, cross_attention_dim=32,
        attention_head_dim=8, use_linear_projection=True, norm_num_groups=32,
        in_channels=4, out_channels=4, mid_block_type=None,
    ).to(DEV)


def test_mid_block_none_in_darkdiff():
    model = DarkDiff(tiny_distilled_unet(), tiny_rvae(), DDPMScheduler(prediction_type="epsilon"),
                     torch.randn(1, 77, 32, device=DEV), region_size=8).train()
    noisy = torch.randn(2, 3, 64, 64, device=DEV)
    clean = torch.randn(2, 3, 64, 64, device=DEV)
    srgb = torch.rand(2, 3, 64, 64, device=DEV) * 2 - 1
    loss, logs = model.training_loss(noisy, clean, srgb)
    loss.backward()
    assert torch.isfinite(loss), logs
    model.eval()
    out = model.sample(torch.rand(1, 3, 64, 64, device=DEV) * 2 - 1, steps=2, guidance=2.0)
    assert out.shape == (1, 3, 64, 64), out.shape


if __name__ == "__main__":
    print("device:", DEV)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
