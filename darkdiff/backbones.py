"""Diffusion U-Net backbones for DarkDiff: pretrained SD distillations and compact U-Nets trained from scratch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D

from darkdiff.model import SD_MODEL, empty_prompt_embeds, load_pretrained

PRETRAINED = {
    "sd21": SD_MODEL,
    "bk-sdm-v2-base": "nota-ai/bk-sdm-v2-base",
    "bk-sdm-v2-small": "nota-ai/bk-sdm-v2-small",
    "bk-sdm-v2-tiny": "nota-ai/bk-sdm-v2-tiny",
    "small-sd": "segmind/small-sd",
    "tiny-sd": "segmind/tiny-sd",
}
SCRATCH = ("compact-unet", "mobilenetv3-unet")
BACKBONES = (*PRETRAINED, *SCRATCH)
BASELINE = "sd21"  # paper DarkDiff, run vanilla; every lightweight backbone runs with the illumination filters


def variant(backbone):
    return "vanilla" if backbone == BASELINE else "illum"
CROSS_DIM = 1024  # SD2.1 OpenCLIP text width; the scratch U-Nets reuse SD2.1's empty-prompt embedding


def compact_unet():
    # SD2.1 layout at half width with one layer per block. attention_head_dim is the head count (SD2 quirk).
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        block_out_channels=(160, 320, 640, 640), layers_per_block=1, cross_attention_dim=CROSS_DIM,
        attention_head_dim=(5, 10, 20, 20), use_linear_projection=True, in_channels=4, out_channels=4,
    )


class MBConvBlock(nn.Module):
    """MobileNetV3 inverted residual with timestep injection. Drop-in for ResnetBlock2D."""

    def __init__(self, in_ch, out_ch, temb_ch, expand=2):
        super().__init__()
        # width from out_ch: up-block inputs are skip concats (2x out), in_ch-based width outgrows ResnetBlock2D
        hid = out_ch * expand
        self.in_channels, self.out_channels = in_ch, out_ch
        self.norm = nn.GroupNorm(math.gcd(32, in_ch), in_ch)
        self.expand = nn.Conv2d(in_ch, hid, 1)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.temb = nn.Linear(temb_ch, hid)
        self.se_reduce = nn.Conv2d(hid, hid // 4, 1)
        self.se_expand = nn.Conv2d(hid // 4, hid, 1)
        self.project = nn.Conv2d(hid, out_ch, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, input_tensor, temb, *args, **kwargs):  # diffusers calls resnets as (x, temb)
        h = F.hardswish(self.expand(self.norm(input_tensor)))
        h = F.hardswish(self.dw(h) + self.temb(F.silu(temb))[:, :, None, None])
        s = F.hardsigmoid(self.se_expand(F.relu(self.se_reduce(F.adaptive_avg_pool2d(h, 1)))))
        return self.skip(input_tensor) + self.project(s * h)


def mobilenetv3_unet():
    unet = compact_unet()
    swaps = [(m, n, c) for m in unet.modules() for n, c in m.named_children() if isinstance(c, ResnetBlock2D)]
    for parent, name, r in swaps:
        setattr(parent, name, MBConvBlock(r.in_channels, r.out_channels, r.time_emb_proj.in_features))
    return unet


def load_backbone(name, weights=True):
    """Return (unet, scheduler, prompt_embeds). weights=False builds random U-Nets from JSON configs only."""
    if name in SCRATCH:
        unet = compact_unet() if name == "compact-unet" else mobilenetv3_unet()
        # SD's scheduler settings. clip_sample must be off: the diffusers default clips sampled latents to [-1, 1]
        sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085,
                              beta_end=0.012, prediction_type="epsilon", clip_sample=False, steps_offset=1)
        embeds = empty_prompt_embeds(SD_MODEL) if weights else torch.zeros(1, 77, CROSS_DIM)
        return unet, sched, embeds

    model_id = PRETRAINED[name]
    if weights:
        unet, sched = load_pretrained(model_id)
        return unet, sched, empty_prompt_embeds(model_id)
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(model_id, subfolder="unet"))
    sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return unet, sched, torch.zeros(1, 77, unet.config.cross_attention_dim)
