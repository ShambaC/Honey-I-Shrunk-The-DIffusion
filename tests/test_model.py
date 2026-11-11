"""Self-check for darkdiff.model with tiny random SD2-shaped U-Net and VAE. Run: uv run tests/test_model.py"""
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention

from darkdiff.model import DarkDiff, RegionCrossAttention
from darkdiff.vae import ResidualVAE

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_unet():
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        block_out_channels=(32, 64, 64, 64), layers_per_block=1, cross_attention_dim=32,
        attention_head_dim=8, use_linear_projection=True, norm_num_groups=32,
        in_channels=4, out_channels=4,
    ).to(DEV)


def tiny_rvae():
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(32, 32, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        layers_per_block=1, norm_num_groups=32, sample_size=64,
    )
    return ResidualVAE(vae).to(DEV)


def make(unet=None, illum=False):
    unet = unet or tiny_unet()
    sched = DDPMScheduler(prediction_type="v_prediction")
    pe = torch.randn(1, 77, 32, device=DEV)
    return DarkDiff(unet, tiny_rvae(), sched, pe, region_size=8, illum=illum)


def nonzero_region_out(model, std=0.05):
    for r in model._regions.values():
        nn.init.normal_(r.attn.to_out[0].weight, std=std)


class Zero(nn.Module):
    def forward(self, h, **kw):
        return torch.zeros_like(h)


def test_zero_init_matches_unet_without_encoder_text_attn():
    unet = tiny_unet().eval()
    ref = copy.deepcopy(unet).eval()
    # Replacing encoder text attn2 removes its contribution by design, so the reference
    # has encoder attn2 outputs zeroed. Zero-init region attn must then match exactly.
    blocks = [b for n, b in ref.named_modules()
              if isinstance(b, BasicTransformerBlock) and n.split(".")[0] in ("down_blocks", "mid_block")]
    for b in blocks:
        b.attn2 = Zero()
    m = DarkDiff(unet, tiny_rvae(), DDPMScheduler(prediction_type="v_prediction"),
                 torch.randn(1, 77, 32, device=DEV), region_size=8).eval()
    z = torch.randn(2, 4, 8, 8, device=DEV)
    t = torch.tensor([10, 500], device=DEV)
    with torch.no_grad():
        out = m.denoise(z, t, m.context(torch.randn_like(z)))
        exp = ref(z, t, encoder_hidden_states=m.prompt_embeds.expand(2, -1, -1)).sample
    assert torch.allclose(out, exp, atol=1e-4), (out - exp).abs().max()


def test_region_locality():
    C, r = 32, 4
    rca = RegionCrossAttention(Attention(query_dim=C, heads=8, dim_head=4),
                               Attention(query_dim=C, cross_attention_dim=C, heads=8, dim_head=4), r).to(DEV)
    nn.init.normal_(rca.attn.to_out[0].weight, std=0.1)
    rca.hw = (8, 8)
    x = torch.randn(1, 64, C, device=DEV)
    cond = torch.randn(1, 64, C, device=DEV)
    rca.cond = cond
    with torch.no_grad():
        o1 = rca(x)
        cond2 = cond.clone().view(1, 8, 8, C)
        cond2[:, :4, :4] += 1.0  # region 0 only (top-left 4x4)
        rca.cond = cond2.view(1, 64, C)
        o2 = rca(x)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[:4, :4] = True
    m = m.flatten().to(DEV)
    assert (o1[0, m] - o2[0, m]).abs().max() > 1e-3
    assert (o1[0, ~m] - o2[0, ~m]).abs().max() < 1e-5


def test_region_init_from_cross_attn1():
    # SDXS: attn1 is a text cross-attention (K/V from a wider text dim); only shape-matching weights are copied
    C = 32
    a1 = Attention(query_dim=C, cross_attention_dim=48, heads=8, dim_head=4).to(DEV)
    rca = RegionCrossAttention(a1, Attention(query_dim=C, cross_attention_dim=48, heads=8, dim_head=4), 4).to(DEV)
    assert rca.attn.to_k.weight.shape == (C, C)
    assert torch.equal(rca.attn.to_q.weight, a1.to_q.weight)


def test_non_multiple_grid():
    model = make().eval()
    nonzero_region_out(model)
    z = torch.randn(2, 4, 20, 12, device=DEV)
    with torch.no_grad():
        out = model.denoise(z, torch.tensor([3, 7], device=DEV), model.context(z))
    assert out.shape == z.shape, out.shape


def test_training_loss_grads():
    model = make().train()
    nonzero_region_out(model)  # zero to_out would block grads to context on step 0
    noisy = torch.randn(2, 3, 64, 64, device=DEV)
    clean = torch.randn(2, 3, 64, 64, device=DEV)
    srgb = torch.rand(2, 3, 64, 64, device=DEV) * 2 - 1
    loss, logs = model.training_loss(noisy, clean, srgb)
    loss.backward()
    assert isinstance(logs["ldm"], float) and isinstance(logs["image"], float)
    ctx_grad = [p for p in model.ctx_net.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    reg_grad = [p for r in model._regions.values() for p in r.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert ctx_grad, "context processor got no grads"
    assert reg_grad, "region attn got no grads"
    assert all(p.grad is None for p in model.rvae.parameters()), "frozen VAE got grads"
    groups = model.param_groups(2.5e-4, 5e-5)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.unet.parameters())) + len(list(model.ctx_net.parameters()))
    assert not any(k.startswith("rvae.") for k in model.trainable_state_dict())


def test_sample_shape_range():
    model = make().eval()
    nonzero_region_out(model)
    noisy = torch.rand(1, 3, 64, 64, device=DEV) * 2 - 1
    out = model.sample(noisy, steps=2, guidance=2.0)
    assert out.shape == (1, 3, 64, 64), out.shape
    assert out.min() >= -1 and out.max() <= 1


def test_illum_zero_init_matches_vanilla():
    unet = tiny_unet()
    van = make(copy.deepcopy(unet)).eval()
    nonzero_region_out(van)
    ill = make(copy.deepcopy(unet), illum=True).eval()
    missing, unexpected = ill.load_state_dict(van.state_dict(), strict=False)
    ill.prompt_embeds = van.prompt_embeds  # non-persistent buffer, random per make()
    assert not unexpected and missing and all(any(t in k for t in (".film.", ".gate.", "key_bias")) for k in missing), missing
    z = torch.randn(2, 4, 20, 12, device=DEV)  # non-multiple grid exercises the padded keys
    t = torch.tensor([3, 700], device=DEV)
    L = torch.rand(2, 1, 20, 12, device=DEV)
    with torch.no_grad():
        ctx = van.context(z)
        a = van.denoise(z, t, ctx)
        b = ill.denoise(z, t, ctx, L)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()  # zero-init illum branch is an identity


def test_illum_grads_and_sample():
    model = make(illum=True).train()
    nonzero_region_out(model)
    noisy = torch.rand(2, 3, 64, 64, device=DEV) * 2 - 1
    srgb = torch.rand(2, 3, 64, 64, device=DEV) * 2 - 1
    loss, _ = model.training_loss(noisy, noisy, srgb)
    loss.backward()
    regs = list(model._regions.values())
    assert any(r.film.net[-1].weight.grad.abs().sum() > 0 for r in regs), "film got no grads"
    assert any(r.gate[-1].weight.grad.abs().sum() > 0 for r in regs), "gate got no grads"
    assert any(r.key_bias.grad is not None and r.key_bias.grad.abs().sum() > 0 for r in regs), "key_bias got no grads"
    ids = {id(p) for p in model.param_groups(2.5e-4, 5e-5)[0]["params"]}
    assert all(id(r.key_bias) in ids and id(r.film.net[-1].weight) in ids for r in regs), "illum params not in lr_new group"

    model.eval()
    for r in regs:  # make the illum branch non-trivial so sampling exercises it
        nn.init.normal_(r.gate[-1].weight, std=0.1)
        nn.init.normal_(r.key_bias, std=0.5)
    out = model.sample(noisy[:1], steps=2, guidance=2.0)
    assert out.shape == (1, 3, 64, 64) and torch.isfinite(out).all()


if __name__ == "__main__":
    print("device:", DEV)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
