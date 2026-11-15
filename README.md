<div align="center">

# Honey, I Shrunk The Diffusion

### _Lightweight, illumination-aware DarkDiff for low-light raw enhancement_

![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20%7C%20CUDA%2013-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/🤗%20diffusers-%E2%89%A50.42-FFD21E)
![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9)
![Course](https://img.shields.io/badge/course-AI61201-blue)
![Status](https://img.shields.io/badge/status-research%20code-orange)

_A course project for **Visual Computing with AI/ML** (AI61201)_
_by **Shamba Chowdhury** (lightweight backbones) and **Ahon Bandyopadhyay** (illumination-aware conditioning)_

</div>

---

An unofficial PyTorch implementation of [_DarkDiff: Advancing Low-Light Raw Enhancement by Retasking Diffusion Models for Camera ISP_](https://arxiv.org/abs/2505.23743) (PDF in `papers/`), extended in two ways:

1. **Lightweight backbones.** _Swap the 866M SD 2.1 U-Net for smaller pretrained diffusion U-Nets (327M to 584M)._
2. **Illumination-aware conditioning.** _An illumination map from the noisy input modulates the region cross-attention (`--illum`)._

---

## 🚀 Quick start

_Use `uv` only: `uv run script.py`, `uv add pkg`. Every script takes `--gpu N`._

### 1. Setup and data

```sh
uv sync
uv run download_data.py --root data --datasets sid eld lrd
```

### 2. Train

```sh
# one backbone, with illumination conditioning
uv run train_ldm.py --dataset sid --backbone bk-sdm-v2-tiny --illum \
  --data-root data --epochs 10 --out runs/ldm

# the paper baseline (vanilla, no --illum)
uv run train_ldm.py --dataset sid --backbone sd21 --out runs/ldm_sd21
```

_Checkpoints are written to `--out` (`ldm_last.pt`, plus one every `--save-every` epochs). They hold weights only: there is no resume._

_Optional stage 1 (`train_vae.py`, then pass `--vae-ckpt runs/vae/vae_last.pt` everywhere) is slow, about 2 h/epoch on an A40. Without it, the pretrained SD VAE is used._

### 3. Evaluate

```sh
# PSNR / SSIM / LPIPS on full SID test images, writes runs/eval/metrics.json
uv run evaluate.py --dataset sid --backbone bk-sdm-v2-tiny --illum \
  --ckpt runs/ldm/ldm_last.pt --data-root data --save-images

# ELD, using the SID-trained model (guidance 2.5)
uv run evaluate.py --dataset eld --backbone bk-sdm-v2-tiny --illum \
  --ckpt runs/ldm/ldm_last.pt --data-root data
```

> ⚠️ _A checkpoint trained with `--illum` must be evaluated with `--illum`, and with the same `--backbone`._

### 4. Inference on a few images

```sh
# noisy | output | ground-truth PNGs in runs/samples/<backbone>_<dataset>
uv run sample_images.py --backbone bk-sdm-v2-tiny --dataset sid -n 8 \
  --ckpt runs/ldm/ldm_last.pt --data-root data
```

_`--ckpt` defaults to where the sweep saved that backbone. Illum or vanilla is read from the checkpoint keys._

### 5. Inference notebook (Kaggle)

[Kaggle notebook](https://www.kaggle.com/code/shambac/sample-inference-histd)

### 6. Compare backbones (server)

```sh
uv run sweep_backbones.py --epochs 10 --backbones sd21 bk-sdm-v2-tiny tiny-sd sdxs-512 \
  --data-root /path/to/DarkDiff_Data --gpu 1
```

_Trains, evaluates and benchmarks each backbone, then writes `runs/backbones/summary.csv`. See [Sweep output](#-sweep-output)._

---

## 🧭 Pipeline

| Stage                      | What happens                                                                                                                                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Input**                  | Raw → minimal ISP (`darkdiff/isp.py`): Bayer pack to half resolution, black/white normalise, × exposure ratio, white balance from the ground truth, average the greens → _noisy LRGB_. Target: clean LRGB → CCM → sRGB gamma. Tensors in [−1, 1]. |
| **Condition**              | Frozen `ResidualVAE` (`darkdiff/vae.py`): SD VAE plus zero-init encoder→decoder skip convs. `z_y = E(noisy)`.                                                                                                                                     |
| **Context processor**      | A copy of the U-Net encoder, run at t=0 with the empty prompt, _once per image_.                                                                                                                                                                  |
| **Region cross-attention** | 8×8 token windows. Replaces text cross-attention in the down and mid blocks. Q/K/V copied from `attn1`, `to_out` zero-init. Up blocks keep text cross-attention with the empty prompt.                                                            |
| **Loss**                   | `L_LDM` (ε or v-prediction) + λ · `L_image`, where `L_image` is the MSE of `D(x̂₀)` against the sRGB target through the frozen decoder. 5% of conditions are replaced by Gaussian noise for CFG.                                                   |
| **Sampling**               | DDIM, 50 steps, guidance 2.0 (SID / LRD) or 2.5 (ELD). The unconditional branch uses a noise condition.                                                                                                                                           |

---

## 🪶 Backbones

_All are pretrained in the SD latent space, as DarkDiff requires. Pick one with `--backbone`._

| Name              | Source                               | U-Net | Pred. | Notes                                                                         |
| ----------------- | ------------------------------------ | ----- | ----- | ----------------------------------------------------------------------------- |
| `sd21`            | `sd2-community/stable-diffusion-2-1` | 866M  | v     | _Paper baseline (the original `stabilityai` repo is gone)_                    |
| `bk-sdm-v2-base`  | `nota-ai/bk-sdm-v2-base`             | 584M  | ε     | _Distilled from SD 2.1-base_                                                  |
| `bk-sdm-v2-small` | `nota-ai/bk-sdm-v2-small`            | 486M  | ε     |                                                                               |
| `bk-sdm-v2-tiny`  | `nota-ai/bk-sdm-v2-tiny`             | 327M  | ε     |                                                                               |
| `small-sd`        | `segmind/small-sd`                   | 582M  | ε     | _SD 1.5 family, text width 768_                                               |
| `tiny-sd`         | `segmind/tiny-sd`                    | 324M  | ε     | _SD 1.5 family, text width 768_                                               |
| `sdxs-512`        | `IDKiro/sdxs-512-0.9`                | 328M  | ε     | _1-step distilled; `attn1` is text cross-attention at its highest resolution_ |

_`sd21` is the baseline and runs vanilla. Every other backbone runs with `--illum`._

> 📝 _Scratch-trained compact and MobileNet-style U-Nets were tried and removed: DarkDiff needs a pretrained prior, and no public pretrained compact or MobileNet diffusion U-Net exists (SnapFusion, MobileDiffusion and SnapGen are unreleased). They are in git history before `a32e354`._

---

## 💡 Illumination-aware cross-attention

_`--illum` adds three zero-initialised pieces inside every region cross-attention layer, driven by φ(L) = [L, log L]:_

- **Illumination map.** `L` is the luminance of the noisy LRGB, average-pooled to the latent grid, clamped and smoothed (`estimate_illumination`).
- **`IllumFiLM`** modulates the condition before it becomes keys and values.
- **Key bias** `b_h · log L` per head lets queries down-weight dark, low-SNR condition tokens.
- **Output gate** `1 + tanh(G(φ(L)))` sets how much each region trusts the condition.

_Everything starts at zero, so the model is exactly vanilla DarkDiff at step 0. Dropped and unconditional conditions use `L = 1`._

_`CurveBrighten`, `LightContextEncoder` and `illum_adaptive_cfg` also live in `darkdiff/illum.py` but are **not wired in** (`uv run -m darkdiff.illum` runs their self-test)._

---

## 📄 Report

_The project report is an IEEE-format LaTeX paper in [`report/`](report/)._

| Path                                | Content                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `report/main.tex`                   | _Entry point_, loads `macros.tex` and the sections                                                    |
| `report/sections/`                  | `abstract`, `introduction`, `related_work`, `methodology`, `experiments`, `conclusion`, `appendix_contributions` |
| `report/figures/`                   | Pipeline diagram (`pipeline.tex`) and qualitative results: input, reference, `sd21`, `bk-sdm-v2-tiny`, `tiny-sd`, `sdxs-512` |
| `report/darkdiff_refs.bib`          | Bibliography                                                                                          |
| `report/build/`                     | _LaTeX build output, git-ignored_                                                                     |

```sh
cd report
latexmk -pdf -outdir=build main.tex   # writes report/build/main.pdf
```

---

## 🗂️ Layout

| File                     | Role                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `darkdiff/isp.py`        | _Sec 4.1 / A2:_ minimal ISP, LRGB input and sRGB target                                    |
| `darkdiff/data.py`       | SID (Sony, ×300 test), ELD (SonyA7S2, ×200), LRD (−3EV) pair lists + `PairDataset`         |
| `darkdiff/vae.py`        | _Sec 4.3:_ residual VAE + LDM stage-1 loss (L1 + LPIPS + PatchGAN + KL)                    |
| `darkdiff/backbones.py`  | Pretrained U-Net registry (`--backbone`)                                                   |
| `darkdiff/illum.py`      | _Extension:_ illumination map, FiLM, key bias, output gate (`--illum`)                     |
| `darkdiff/model.py`      | _Sec 4.2 / 4.4:_ context processor, region cross-attention, image loss, CFG sampling       |
| `train_vae.py`           | Stage 1 (optional): noisy LRGB → clean sRGB VAE fine-tune                                  |
| `train_ldm.py`           | Stage 2: U-Net + context processor + region attention                                      |
| `evaluate.py`            | DDIM 50 steps, PSNR / SSIM / LPIPS on full images, writes `metrics.json`                   |
| `sample_images.py`       | Noisy, output and ground-truth PNGs for N distinct test scenes; prints PSNR / SSIM / LPIPS |
| `benchmark_backbones.py` | Params, GFLOPs, time and memory with random weights                                        |
| `sweep_backbones.py`     | Train + evaluate + benchmark every backbone, write `summary.csv`                           |
| `download_data.py`       | SID / ELD / LRD download + extract                                                         |
| `tests/`                 | Tiny-random-model tests, no downloads                                                      |
| `report/`                | _LaTeX project report_, see [Report](#-report)                                             |

---

## 📊 Sweep output

_`sweep_backbones.py` writes `runs/backbones/sd21` and `runs/backbones/<backbone>-illum`, then `runs/backbones/summary.csv`:_

- One row per backbone: PSNR / SSIM / LPIPS for `sid` and `eld` (or `lrd`), plus benchmark columns (`unet_params_M`, `ctx_params_M`, `trainable_params_M`, `unet_gflops`, `ctx_gflops`, `infer_s`, `infer_mem_GB`, `train_step_s`, `train_mem_GB`).
- The first row is `how to read`: _it says per column whether higher or lower is better._
- `*_delta` columns are the row minus `sd21`.
- `pareto` is True when no other backbone has lower-or-equal LPIPS and lower-or-equal `trainable_params_M`, with one strictly lower.
- `OOM` in a benchmark column marks a run that does not fit the GPU.

_An existing `ldm_last.pt` makes the sweep treat that run as done._

_Efficiency only, no data or checkpoints needed:_ `uv run benchmark_backbones.py`

---

## 🧪 Tests

_Tiny random models, no downloads:_

```sh
uv run tests/test_isp.py
uv run tests/test_vae.py
uv run tests/test_model.py
uv run tests/test_backbones.py
uv run tests/test_benchmark.py
uv run tests/test_smoke.py
```

---

## 🧩 Choices the paper does not specify

- _Residual VAE skip convs and region-attention output projections are zero-init; Q/K/V are copied from the block's self-attention (shape-matching weights only)._
- _Region size is 8×8 latent tokens at every layer (`--region-size`)._
- _The context processor runs at timestep 0, once per image, and is reused at every sampling step._
- _The CFG unconditional branch uses a Gaussian-noise condition, matching the 5% training drop._
- _`λ = 1.0`._
- _ELD uses a fixed ×200 ratio instead of the EXIF-derived one. LRD ratios come from filenames and are unverified._
- _Optimiser: Adam, lr 2.5e-4 for the context processor + region attention, 5e-5 for the rest of the U-Net, bf16 autocast, batch 1, 1200 crops. The VAE decoder uses gradient checkpointing when grads are on._

_Defaults follow paper A2 (3000 epochs, 8× A100). Our runs use far fewer epochs on one GPU (A40 48 GB; a local RTX 5070 12 GB for smoke tests only)._

---

## ⚠️ Known issues

1. **Target-latent mismatch without a stage-1 VAE.** `training_loss` uses `z0 = E(clean LRGB)`; the plain SD VAE decodes that to linear RGB, but `L_image` targets sRGB. _Proposed fix: use `z0 = E(srgb)` when `--vae-ckpt` is omitted._
2. **`sd21` is unstable at 10 epochs.** Outputs look like undenoised noise (`bk-sdm-v2-tiny` is blurry but clean). _Likely cause: 1.22B trainable params, batch 1, no warmup, no grad clipping. Proposed: `clip_grad_norm_(..., 1.0)` + lr warmup.\_
3. **Backbone and illum are confounded** in the sweep (`sd21` vanilla vs lightweight + illum). _An ablation, e.g. `bk-sdm-v2-tiny` with and without `--illum`, is still needed._
4. **No resume.** _Checkpoints are weights only._

---

<div align="center">

_Built on [DarkDiff](https://arxiv.org/abs/2505.23743) and [Stable Diffusion](https://huggingface.co/sd2-community/stable-diffusion-2-1). This is an unofficial reimplementation._

</div>
