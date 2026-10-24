import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


class ResidualVAE(nn.Module):
    """SD VAE with encoder->decoder residual skips (DarkDiff Sec 4.3, Fig 7)."""

    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
        n = len(vae.config.block_out_channels)
        # skips[b]: Conv_b maps encoder feature E_b to the input of decoder.up_blocks[n-1-b].
        self.skips = nn.ModuleList()
        for b in range(n):
            up_in = vae.decoder.up_blocks[n - 1 - b].resnets[0].conv1.in_channels
            conv = nn.Conv2d(vae.config.block_out_channels[b], up_in, 3, padding=1)
            # zero init so the wrapper starts identical to the pretrained VAE (paper does not specify init)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.skips.append(conv)

    def encode(self, x):
        assert x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0, "H, W must be multiples of 8"
        enc = self.vae.encoder
        h = enc.conv_in(x)
        feats = []
        for down in enc.down_blocks:
            # shortcut: assumes DownEncoderBlock2D (resnets + downsamplers only), as in the SD VAE
            for resnet in down.resnets:
                h = resnet(h, temb=None)
            feats.append(h)
            if down.downsamplers is not None:
                for ds in down.downsamplers:
                    h = ds(h)
        h = enc.mid_block(h)
        h = enc.conv_out(enc.conv_act(enc.conv_norm_out(h)))
        if self.vae.quant_conv is not None:
            h = self.vae.quant_conv(h)
        return DiagonalGaussianDistribution(h), feats

    def decode(self, z, feats):
        dec = self.vae.decoder
        h = self.vae.post_quant_conv(z) if self.vae.post_quant_conv is not None else z
        h = dec.mid_block(dec.conv_in(h))
        n = len(self.skips)
        for j, up in enumerate(dec.up_blocks):
            b = n - 1 - j
            h = up(h + self.skips[b](feats[b]))
        return dec.conv_out(dec.conv_act(dec.conv_norm_out(h)))

    def forward(self, x, sample_posterior=True):
        posterior, feats = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z, feats), posterior

    def last_layer(self):
        return self.vae.decoder.conv_out.weight

    @classmethod
    def from_pretrained(cls, model_id):
        return cls(AutoencoderKL.from_pretrained(model_id, subfolder="vae"))


class NLayerDiscriminator(nn.Module):
    """pix2pix PatchGAN, as in LDM taming/ldm."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        norm = nn.BatchNorm2d
        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = 1, 1
        for n in range(1, n_layers):
            mult_prev, mult = mult, min(2**n, 8)
            seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=2, padding=padw, bias=False),
                    norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        mult_prev, mult = mult, min(2**n_layers, 8)
        seq += [nn.Conv2d(ndf * mult_prev, ndf * mult, kw, stride=1, padding=padw, bias=False),
                norm(ndf * mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf * mult, 1, kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*seq)

    def forward(self, x):
        return self.main(x)


def _weights_init(m):
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class VAELoss(nn.Module):
    """LDM LPIPSWithDiscriminator: L1 + LPIPS nll, KL, adaptive-weighted patch GAN."""

    def __init__(self, disc_start=50001, kl_weight=1e-6, disc_weight=0.5,
                 perceptual_weight=1.0, disc_factor=1.0):
        super().__init__()
        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.perceptual_weight = perceptual_weight
        self.disc_factor = disc_factor
        self.logvar = nn.Parameter(torch.zeros(()))
        self.lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.disc = NLayerDiscriminator().apply(_weights_init)

    def train(self, mode=True):
        super().train(mode)
        self.lpips.eval()  # frozen
        return self

    def _factor(self, step):
        return self.disc_factor if step >= self.disc_start else 0.0

    def _adaptive_weight(self, nll, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        w = nll_grads.norm() / (g_grads.norm() + 1e-4)
        return (w.clamp(0.0, 1e4) * self.disc_weight).detach()

    def generator_loss(self, target, rec, posterior, step, last_layer):
        rec_loss = (target - rec).abs() + self.perceptual_weight * self.lpips(target, rec)
        nll = rec_loss / torch.exp(self.logvar) + self.logvar
        nll = nll.sum() / nll.shape[0]
        kl = posterior.kl().mean()
        g_loss = -self.disc(rec).mean()
        factor = self._factor(step)
        d_weight = self._adaptive_weight(nll, g_loss, last_layer) if factor > 0 else 0.0
        loss = nll + self.kl_weight * kl + d_weight * factor * g_loss
        logs = {"nll": nll.item(), "kl": kl.item(), "g": g_loss.item(), "d_weight": float(d_weight)}
        return loss, logs

    def discriminator_loss(self, target, rec, step):
        real = self.disc(target.detach())
        fake = self.disc(rec.detach())
        hinge = 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())
        loss = self._factor(step) * hinge
        return loss, {"d_hinge": hinge.item()}
