"""DarkDiff core: SD2.1 U-Net + context processor + region-based cross-attention (paper Sec 4.2)."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from tqdm import tqdm

SD_MODEL = "sd2-community/stable-diffusion-2-1"
ENC = ("down_blocks", "mid_block")  # decoder (up_blocks) keeps its text cross-attention


def _enc_sites(unet):
    """[(attn2 module path, BasicTransformerBlock)] for the encoder-side transformer blocks."""
    return [(f"{n}.attn2", m) for n, m in unet.named_modules()
            if isinstance(m, BasicTransformerBlock) and m.attn2 is not None and n.split(".")[0] in ENC]


def _set_grid(module, args):
    # Transformer2DModel input is (B, C, H, W); region attention needs the token grid.
    for m in module.modules():
        if isinstance(m, RegionCrossAttention):
            m.hw = tuple(args[0].shape[-2:])


def _capture(store, name):
    def hook(module, args, kwargs):
        store[name] = args[0] if args else kwargs["hidden_states"]
    return hook


class RegionCrossAttention(nn.Module):
    """attn2 replacement: each r x r window attends only to the context tokens of the same window."""

    def __init__(self, attn1, attn2, region_size):
        super().__init__()
        C = attn1.to_q.in_features
        self.r = region_size
        self.attn = Attention(query_dim=C, cross_attention_dim=C, heads=attn2.heads,
                              dim_head=attn2.to_q.out_features // attn2.heads).to(attn1.to_q.weight)
        self.attn.load_state_dict(attn1.state_dict())  # init from self-attention
        # Paper does not specify init. Zero output projection keeps the region branch silent at start.
        nn.init.zeros_(self.attn.to_out[0].weight)
        nn.init.zeros_(self.attn.to_out[0].bias)
        self.cond = None  # (B, N, C) context tokens, set by DarkDiff.denoise
        self.hw = None  # (H, W) token grid, set by _set_grid

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Text encoder_hidden_states / mask are unused: the condition comes from self.cond.
        B, N, C = hidden_states.shape
        H, W = self.hw
        r = self.r
        Kh, Kw = -(-H // r), -(-W // r)
        Hp, Wp = Kh * r, Kw * r

        def windows(x):  # (B, N, C) -> (B*K, r*r, C), zero-padded to whole regions
            x = F.pad(x.reshape(B, H, W, C), (0, 0, 0, Wp - W, 0, Hp - H))
            return x.reshape(B, Kh, r, Kw, r, C).permute(0, 1, 3, 2, 4, 5).reshape(B * Kh * Kw, r * r, C)

        valid = torch.zeros(1, Hp, Wp, dtype=torch.bool, device=hidden_states.device)
        valid[:, :H, :W] = True  # padded key tokens get masked out
        valid = valid.reshape(1, Kh, r, Kw, r).permute(0, 1, 3, 2, 4).reshape(1, Kh * Kw, 1, r * r)
        mask = valid.expand(B, -1, -1, -1).reshape(B * Kh * Kw, 1, 1, r * r)

        a, h = self.attn, self.attn.heads

        def heads(t):
            return t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2)

        q = heads(a.to_q(windows(hidden_states)))
        k = heads(a.to_k(windows(self.cond)))
        v = heads(a.to_v(windows(self.cond)))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = a.to_out[0](o.transpose(1, 2).reshape(B * Kh * Kw, r * r, -1))
        o = o.reshape(B, Kh, Kw, r, r, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        return o[:, :H, :W].reshape(B, N, C)  # no residual: BasicTransformerBlock adds it


class DarkDiff(nn.Module):
    def __init__(self, unet, rvae, scheduler, prompt_embeds, region_size=8, cfg_drop=0.05):
        super().__init__()
        self.rvae = rvae.requires_grad_(False).eval()
        self.scheduler = scheduler
        self.cfg_drop = cfg_drop
        self.sf = rvae.vae.config.scaling_factor
        self.register_buffer("prompt_embeds", prompt_embeds, persistent=False)

        # Context processor: U-Net encoder copy, applied to z_y at fixed timestep 0.
        self.ctx_net = copy.deepcopy(unet)
        for n in ("up_blocks", "conv_norm_out", "conv_out"):
            delattr(self.ctx_net, n)
        self._cap = {}
        for name, blk in _enc_sites(self.ctx_net):
            blk.attn2.register_forward_pre_hook(_capture(self._cap, name), with_kwargs=True)

        # Main U-Net: encoder attn2 -> region cross-attention (must run after the deepcopy).
        self.unet = unet
        self._regions = {}
        for name, blk in _enc_sites(unet):
            blk.attn2 = RegionCrossAttention(blk.attn1, blk.attn2, region_size)
            self._regions[name] = blk.attn2
        for n, m in unet.named_modules():
            if isinstance(m, Transformer2DModel) and n.split(".")[0] in ENC:
                m.register_forward_pre_hook(_set_grid)

    def train(self, mode=True):
        super().train(mode)
        self.rvae.eval()  # frozen VAE stays in eval
        return self

    @torch.no_grad()
    def condition(self, noisy):
        posterior, feats = self.rvae.encode(noisy)
        return posterior.mode() * self.sf, feats

    def context(self, z_y):
        """{attn2 path: (B, N, C)} context tokens; timestep fixed to 0, empty prompt."""
        net = self.ctx_net
        B = z_y.shape[0]
        t = torch.zeros(B, dtype=torch.long, device=z_y.device)
        enc = self.prompt_embeds.expand(B, -1, -1)
        emb = net.time_embedding(net.time_proj(t).to(z_y.dtype))
        x = net.conv_in(z_y)
        self._cap.clear()
        for blk in net.down_blocks:
            kw = {"encoder_hidden_states": enc} if getattr(blk, "has_cross_attention", False) else {}
            x, _ = blk(hidden_states=x, temb=emb, **kw)
        if net.mid_block is not None:  # distilled U-Nets (bk-sdm-*-small/tiny) have no mid block
            net.mid_block(x, emb, encoder_hidden_states=enc)
        return {n: self._cap.pop(n) for n in self._regions}

    def denoise(self, z_t, t, ctx):
        for name, reg in self._regions.items():
            reg.cond = ctx[name]
        enc = self.prompt_embeds.expand(z_t.shape[0], -1, -1)
        return self.unet(z_t, t, encoder_hidden_states=enc).sample

    def _target(self, z0, noise, t):
        pt = self.scheduler.config.prediction_type
        if pt == "epsilon":
            return noise
        if pt == "v_prediction":
            return self.scheduler.get_velocity(z0, noise, t)
        raise ValueError(pt)

    def _x0(self, zt, pred, t):
        ab = self.scheduler.alphas_cumprod.to(zt)[t].view(-1, 1, 1, 1)
        pt = self.scheduler.config.prediction_type
        if pt == "epsilon":
            return (zt - (1 - ab).sqrt() * pred) / ab.sqrt()
        if pt == "v_prediction":
            return ab.sqrt() * zt - (1 - ab).sqrt() * pred
        raise ValueError(pt)

    def training_loss(self, noisy, clean, srgb, lam=1.0):
        z_y, feats = self.condition(noisy)
        with torch.no_grad():
            z0 = self.rvae.encode(clean)[0].mode() * self.sf
        B = z0.shape[0]
        drop = torch.rand(B, device=z0.device) < self.cfg_drop  # CFG: drop condition (paper A2)
        z_y = torch.where(drop.view(B, 1, 1, 1), torch.randn_like(z_y), z_y)

        noise = torch.randn_like(z0)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=z0.device)
        zt = self.scheduler.add_noise(z0, noise, t)
        pred = self.denoise(zt, t, self.context(z_y))
        ldm = F.mse_loss(pred, self._target(z0, noise, t))

        # One-step x0 estimate (paper Eq 8); gradients flow through the frozen decoder (Eq 9).
        img = self.rvae.decode(self._x0(zt, pred, t) / self.sf, feats)
        image = F.mse_loss(img, srgb)
        return ldm + lam * image, {"ldm": ldm.item(), "image": image.item()}

    @torch.no_grad()
    def sample(self, noisy, steps=50, guidance=2.0):
        z_y, feats = self.condition(noisy)
        ctx_c = self.context(z_y)
        ctx_u = self.context(torch.randn_like(z_y))  # unconditional branch, as in CFG training
        ctx = {k: torch.cat([ctx_u[k], ctx_c[k]]) for k in ctx_c}

        sched = DDIMScheduler.from_config(self.scheduler.config)
        sched.set_timesteps(steps, device=z_y.device)
        z = torch.randn_like(z_y)
        for t in tqdm(sched.timesteps, desc="sample"):
            eu, ec = self.denoise(torch.cat([z, z]), t, ctx).chunk(2)
            z = sched.step(eu + guidance * (ec - eu), t, z).prev_sample
        return self.rvae.decode(z / self.sf, feats).clamp(-1, 1)

    def param_groups(self, lr_new, lr_unet):
        new = list(self.ctx_net.parameters()) + [p for r in self._regions.values() for p in r.parameters()]
        ids = {id(p) for p in new}
        rest = [p for p in self.unet.parameters() if id(p) not in ids]
        return [{"params": new, "lr": lr_new}, {"params": rest, "lr": lr_unet}]

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("rvae.")}


@torch.no_grad()
def empty_prompt_embeds(model_id=SD_MODEL):
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt")
    out = enc(ids.input_ids).last_hidden_state
    del enc  # free the text encoder
    return out


def load_pretrained(model_id=SD_MODEL):
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return unet, sched
