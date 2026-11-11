"""Diffusion U-Net backbones for DarkDiff. All are pretrained in the SD VAE latent space, as the paper requires."""
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from darkdiff.model import SD_MODEL, empty_prompt_embeds, load_pretrained

PRETRAINED = {
    "sd21": SD_MODEL,
    "bk-sdm-v2-base": "nota-ai/bk-sdm-v2-base",
    "bk-sdm-v2-small": "nota-ai/bk-sdm-v2-small",
    "bk-sdm-v2-tiny": "nota-ai/bk-sdm-v2-tiny",
    "small-sd": "segmind/small-sd",
    "tiny-sd": "segmind/tiny-sd",
    # SD2.1-base distilled for 1-step generation (same block layout as bk-sdm-v2-tiny); DarkDiff fine-tunes it for 50-step DDIM
    "sdxs-512": "IDKiro/sdxs-512-0.9",
}
BACKBONES = tuple(PRETRAINED)
BASELINE = "sd21"  # paper DarkDiff, run vanilla; every lightweight backbone runs with the illumination filters


def variant(backbone):
    return "vanilla" if backbone == BASELINE else "illum"


def load_backbone(name, weights=True):
    """Return (unet, scheduler, prompt_embeds). weights=False builds a random U-Net from the JSON config only."""
    model_id = PRETRAINED[name]
    if weights:
        unet, sched = load_pretrained(model_id)
        return unet, sched, empty_prompt_embeds(model_id)
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(model_id, subfolder="unet"))
    sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return unet, sched, torch.zeros(1, 77, unet.config.cross_attention_dim)
