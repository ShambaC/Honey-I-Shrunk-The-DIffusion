import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import AutoencoderKL

from darkdiff.vae import ResidualVAE, VAELoss

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_vae():
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(32, 64, 64, 64),
        layers_per_block=1, norm_num_groups=32, latent_channels=4,
    )


# 1. zero-init skips: wrapper equals pretrained VAE, feats have expected shapes
vae = tiny_vae().eval().to(dev)
rvae = ResidualVAE(vae).eval().to(dev)
x = torch.randn(2, 3, 64, 64, device=dev)
with torch.no_grad():
    post, feats = rvae.encode(x)
    ref = vae.encode(x).latent_dist
    assert torch.allclose(post.mean, ref.mean, atol=1e-4), "posterior mean mismatch"
    assert [tuple(f.shape) for f in feats] == [
        (2, 32, 64, 64), (2, 64, 32, 32), (2, 64, 16, 16), (2, 64, 8, 8)]
    z = torch.randn(2, 4, 8, 8, device=dev)
    assert torch.allclose(rvae.decode(z, feats), vae.decode(z).sample, atol=1e-4), "decode mismatch"
    rec, p = rvae(x)
    assert rec.shape == x.shape and p.mean.shape == (2, 4, 8, 8)
print("equivalence OK")

# 2. skips get gradients after one step from nonzero init
rvae = ResidualVAE(tiny_vae()).to(dev).train()
for c in rvae.skips:
    torch.nn.init.normal_(c.weight, std=0.02)
before = [c.weight.detach().clone() for c in rvae.skips]
opt = torch.optim.Adam(rvae.parameters(), lr=1e-3)
target = torch.rand(2, 3, 64, 64, device=dev) * 2 - 1
noisy = (target + 0.1 * torch.randn_like(target)).clamp(-1, 1)
loss_fn = VAELoss(disc_start=0).to(dev)  # disc active from step 0
rec, post = rvae(noisy)
loss, logs = loss_fn.generator_loss(target, rec, post, step=1, last_layer=rvae.last_layer())
assert torch.isfinite(loss), logs
loss.backward()
assert all(c.weight.grad is not None and c.weight.grad.abs().sum() > 0 for c in rvae.skips), "no skip grads"
assert loss_fn.logvar.grad is not None and loss_fn.disc.main[0].weight.grad is not None
opt.step()
opt.zero_grad()
assert all(not torch.equal(b, c.weight.detach()) for b, c in zip(before, rvae.skips)), "skips did not update"
print("skip grads OK", logs)

# 3. discriminator loss backward, and factor 0 before disc_start gives zero loss
rec, post = rvae(noisy)
d_loss, dlogs = loss_fn.discriminator_loss(target, rec, step=1)
assert torch.isfinite(d_loss)
d_loss.backward()
assert loss_fn.disc.main[-1].weight.grad is not None
off = VAELoss(disc_start=100).to(dev)
g_off, _ = off.generator_loss(target, rec.detach(), post, step=1, last_layer=rvae.last_layer())
assert torch.isfinite(g_off)
d0, _ = off.discriminator_loss(target, rec.detach(), step=1)
assert d0.item() == 0.0
print("discriminator OK", dlogs)
print("ALL OK on", dev)
