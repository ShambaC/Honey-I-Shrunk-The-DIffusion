"""
Illumination-aware extensions for a DarkDiff-style low-light diffusion model.

Components
----------
1. estimate_illumination   : noise-robust illumination map from gain-amplified LRGB
2. CurveBrighten           : optional learnable, sign-preserving tone curve on LRGB
                             (pixel-space "brightening filter" before the VAE encoder)
3. IllumFiLM               : feature-space brightening / modulation of condition
                             features BEFORE they become keys/values
4. IllumRegionCrossAttention : DarkDiff region-based cross-attention (Sec. 4.2) +
                             illumination key-bias, illumination output gate,
                             optional shifted windows (removes block seams)
5. LightContextEncoder     : MobileNetV3-style, time-independent replacement for the
                             U-Net-copy context processor (computed once, cached)
6. illum_adaptive_cfg      : spatially varying classifier-free guidance

All new branches are zero-initialised so that, at the start of fine-tuning, the
pretrained Stable Diffusion U-Net behaves exactly as before.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-3


# --------------------------------------------------------------------------- #
# 1. Illumination estimation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_illumination(lrgb: torch.Tensor, out_hw=None, smooth: int = 3) -> torch.Tensor:
    """
    lrgb   : (B,3,H,W) black-level-subtracted, gain-amplified linear RGB (can be < 0).
    out_hw : target size, e.g. latent size (H//8, W//8). Average pooling to the
             latent grid averages 64 pixels -> noise std drops ~8x, so the map is
             much more reliable than a full-res max-RGB estimate (which is biased
             upward by noise).
    Returns (B,1,h,w) illumination in [0,1].
    """
    lum = 0.2126 * lrgb[:, 0:1] + 0.7152 * lrgb[:, 1:2] + 0.0722 * lrgb[:, 2:3]
    if out_hw is not None:
        lum = F.adaptive_avg_pool2d(lum, out_hw)          # average BEFORE clamping
    lum = lum.clamp(0.0, 1.0)                              # keeps zero-mean noise unbiased
    if smooth > 1:
        p = smooth // 2
        lum = F.avg_pool2d(F.pad(lum, (p, p, p, p), mode="replicate"), smooth, stride=1)
    return lum


def illum_features(L: torch.Tensor) -> torch.Tensor:
    """Linear + log illumination (log makes multiplicative exposure differences additive)."""
    return torch.cat([L, torch.log(L + EPS) / math.log(1.0 / EPS)], dim=1)


# --------------------------------------------------------------------------- #
# 2. Pixel-space brightening filter (optional, before VAE encoder)
# --------------------------------------------------------------------------- #
class CurveBrighten(nn.Module):
    """
    Zero-DCE-style quadratic curves LE(x) = x + a*x*(1-x), iterated, with per-pixel
    a-maps predicted from (|x|, L). Applied sign-preservingly so the zero-mean noise
    around black level is not rectified. a-maps are predicted at 1/`down` resolution
    and upsampled (they are smooth), keeping cost tiny on 12 MP inputs.
    Zero-init => identity at start.
    """
    def __init__(self, iters: int = 4, ch: int = 16, down: int = 4):
        super().__init__()
        self.iters, self.down = iters, down
        self.net = nn.Sequential(
            nn.Conv2d(5, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3 * iters, 3, 1, 1), nn.Tanh(),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        s, a = torch.sign(x), x.abs().clamp(0, 1)
        small = F.interpolate(a, scale_factor=1 / self.down, mode="area")
        L = estimate_illumination(small)
        alphas = self.net(torch.cat([small, illum_features(L)], 1))
        alphas = F.interpolate(alphas, size=(H, W), mode="bilinear", align_corners=False)
        for al in alphas.chunk(self.iters, dim=1):
            a = a + al * a * (1 - a)
        return s * a


# --------------------------------------------------------------------------- #
# 3. Feature-space brightening before K/V
# --------------------------------------------------------------------------- #
class IllumFiLM(nn.Module):
    """cond' = cond * (1 + gamma(L)) + beta(L). Zero-init => identity."""
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(hidden, 2 * dim, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        L = F.interpolate(L, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.net(illum_features(L)).chunk(2, dim=1)
        return feat * (1 + gamma) + beta


# --------------------------------------------------------------------------- #
# 4. Illumination-aware region-based cross-attention
# --------------------------------------------------------------------------- #
def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B,C,H,W) -> (B*nW, ws*ws, C); H, W must be divisible by ws."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(-1, ws * ws, C)


def window_reverse(t: torch.Tensor, ws: int, B: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, ws*ws, C) -> (B,C,H,W)."""
    C = t.shape[-1]
    t = t.view(B, H // ws, W // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
    return t.reshape(B, C, H, W)


class IllumRegionCrossAttention(nn.Module):
    """
    x    : U-Net features at this layer           (B,C,H,W)  -> queries
    cond : context-encoder features, same size    (B,C,H,W)  -> keys / values
    L    : illumination map (any resolution)      (B,1,h,w)

    Additions over DarkDiff Eq. (5)-(7):
      * IllumFiLM on cond before K/V  (the "brightening filter before cross-attention")
      * per-head learned key bias  b_h * log(L_key): lets queries down-weight
        very dark (low-SNR) condition tokens. Zero-init.
      * output gate g(L) in (0,2): learns how much to trust the condition vs. the
        generative prior per region (dark -> rely more on prior). Starts at 1.
      * shift=True rolls windows by ws/2 (use on alternate layers) to remove seams.
    """
    def __init__(self, dim: int, heads: int = 8, window: int = 8, shift: bool = False):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.ws, self.shift = dim, heads, window, shift
        self.norm_q, self.norm_c = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_out.weight)      # identity residual at init
        nn.init.zeros_(self.to_out.bias)
        self.film = IllumFiLM(dim)
        self.key_bias = nn.Parameter(torch.zeros(heads))
        self.gate = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 1, 3, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws, h = self.ws, self.heads
        Lf = F.interpolate(L, size=(H, W), mode="bilinear", align_corners=False)
        cond = self.film(cond, Lf)

        # pad to multiple of window (full-image inference at arbitrary sizes)
        ph, pw = (-H) % ws, (-W) % ws
        xq, xc, Lk = x, cond, Lf
        if ph or pw:
            pad = (0, pw, 0, ph)
            xq, xc, Lk = (F.pad(t, pad, mode="replicate") for t in (xq, xc, Lk))
        Hp, Wp = H + ph, W + pw
        if self.shift:
            s = ws // 2
            xq, xc, Lk = (torch.roll(t, shifts=(-s, -s), dims=(2, 3)) for t in (xq, xc, Lk))

        q_tok, c_tok = window_partition(xq, ws), window_partition(xc, ws)
        l_tok = window_partition(Lk, ws)                       # (n, N, 1)
        n, N, _ = q_tok.shape

        def split(t):  # (n,N,C) -> (n,h,N,d)
            return t.view(n, N, h, C // h).transpose(1, 2)

        q = split(self.to_q(self.norm_q(q_tok)))
        cn = self.norm_c(c_tok)
        k, v = split(self.to_k(cn)), split(self.to_v(cn))

        # additive key bias: (n,h,1,N)
        logL = torch.log(l_tok.squeeze(-1) + EPS)              # (n,N)
        bias = self.key_bias.view(1, h, 1, 1) * logL.view(n, 1, 1, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        out = self.to_out(out.transpose(1, 2).reshape(n, N, C))
        out = window_reverse(out, ws, B, Hp, Wp)

        if self.shift:
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(2, 3))
        out = out[..., :H, :W]

        g = 1 + torch.tanh(self.gate(illum_features(Lf)))
        return x + g * out


# --------------------------------------------------------------------------- #
# 5. Lightweight, time-independent context encoder (MobileNetV3-style)
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Hardsigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class InvertedResidual(nn.Module):
    """MobileNetV3 bneck. GroupNorm instead of BatchNorm (diffusion batches are small)."""
    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 4, se: bool = True):
        super().__init__()
        hid = cin * expand
        self.block = nn.Sequential(
            nn.Conv2d(cin, hid, 1, bias=False), nn.GroupNorm(8, hid), nn.Hardswish(),
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False), nn.GroupNorm(8, hid), nn.Hardswish(),
            SqueezeExcite(hid) if se else nn.Identity(),
            nn.Conv2d(hid, cout, 1, bias=False), nn.GroupNorm(8, cout),
        )
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class LightContextEncoder(nn.Module):
    """
    Replaces DarkDiff's context processor (a full copy of the SD U-Net encoder,
    run at every DDIM step). Input: condition latent z_y (4ch) + illumination (2ch).
    No timestep input -> run ONCE per image and cache for all sampling steps.
    Output dims default to SD-2.1 down-block widths at latent /1, /2, /4, /8.
    """
    def __init__(self, latent_ch: int = 4, widths=(64, 128, 256, 256),
                 out_dims=(320, 640, 1280, 1280)):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv2d(latent_ch + 2, w[0], 3, 1, 1), nn.Hardswish())
        self.stages = nn.ModuleList()
        cin = w[0]
        for i, cout in enumerate(w):
            stride = 1 if i == 0 else 2
            self.stages.append(nn.Sequential(
                InvertedResidual(cin, cout, stride),
                InvertedResidual(cout, cout, 1),
            ))
            cin = cout
        self.proj = nn.ModuleList(nn.Conv2d(c, d, 1) for c, d in zip(w, out_dims))

    def forward(self, z_y: torch.Tensor, L: torch.Tensor):
        L = F.interpolate(L, size=z_y.shape[-2:], mode="bilinear", align_corners=False)
        h = self.stem(torch.cat([z_y, illum_features(L)], 1))
        feats = []
        for stage, proj in zip(self.stages, self.proj):
            h = stage(h)
            feats.append(proj(h))
        return feats  # list of (B, out_dims[i], H/2^i, W/2^i)


# --------------------------------------------------------------------------- #
# 6. Illumination-adaptive classifier-free guidance (training-free)
# --------------------------------------------------------------------------- #
def illum_adaptive_cfg(eps_uncond, eps_cond, L, w_bright: float = 1.5, w_dark: float = 3.0,
                       gamma: float = 0.5):
    """
    Spatially varying version of DarkDiff Eq. (10): stronger guidance (more
    generated detail) where it is dark, weaker (more fidelity) where it is bright.
    """
    L = F.interpolate(L, size=eps_cond.shape[-2:], mode="bilinear", align_corners=False)
    darkness = (1 - L.clamp(0, 1)) ** gamma
    w = w_bright + (w_dark - w_bright) * darkness
    return eps_uncond + w * (eps_cond - eps_uncond)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 512, 776                        # deliberately not divisible by 64
    lrgb = torch.randn(B, 3, H, W) * 0.05 + 0.1  # noisy, dark
    lat_hw = (H // 8, W // 8)

    L = estimate_illumination(lrgb, out_hw=lat_hw)
    print("illum", tuple(L.shape), float(L.min()), float(L.max()))

    cb = CurveBrighten()
    y = cb(lrgb)
    print("CurveBrighten identity at init:", torch.allclose(y, lrgb, atol=1e-6) or
          "(differs only where |x|>1 was clamped)", f"{_count(cb):.3f}M params")

    enc = LightContextEncoder()
    z_y = torch.randn(B, 4, *lat_hw)
    feats = enc(z_y, L)
    print("context feats", [tuple(f.shape) for f in feats], f"{_count(enc):.2f}M params")

    for i, (dim, shift) in enumerate([(320, False), (640, True), (1280, False)]):
        attn = IllumRegionCrossAttention(dim, heads=8, window=8, shift=shift)
        x = torch.randn_like(feats[i])
        out = attn(x, feats[i], L)
        assert out.shape == x.shape
        print(f"RbXAttn dim={dim} shift={shift}: out {tuple(out.shape)}, "
              f"identity at init={torch.allclose(out, x)}, {_count(attn):.2f}M params")

    e_u, e_c = torch.randn(B, 4, *lat_hw), torch.randn(B, 4, *lat_hw)
    print("adaptive CFG", tuple(illum_adaptive_cfg(e_u, e_c, L).shape))
