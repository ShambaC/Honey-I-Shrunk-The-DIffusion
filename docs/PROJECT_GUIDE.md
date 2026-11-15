# Project Guide: Lightweight, Illumination-Aware DarkDiff

A study and revision guide for the whole project: the problem, the background theory, the DarkDiff paper, our implementation, our two extensions, the experiment protocol, what went wrong and why, and what to do next.

Course: Visual Computing with AI/ML (AI61201), Autumn 2026-07.
Team: Ahon Bandyopadhyay (illumination-aware conditioning), Shamba Chowdhury (lightweight backbones).
Base paper: *DarkDiff: Advancing Low-Light Raw Enhancement by Retasking Diffusion Models for Camera ISP* (arXiv 2505.23743), in `papers/`.

---

## Contents

1. [The problem](#1-the-problem)
2. [Background: raw images and the camera ISP](#2-background-raw-images-and-the-camera-isp)
3. [Background: diffusion models](#3-background-diffusion-models)
4. [The DarkDiff paper](#4-the-darkdiff-paper)
5. [Our implementation of DarkDiff](#5-our-implementation-of-darkdiff)
6. [Extension 1: lightweight backbones](#6-extension-1-lightweight-backbones)
7. [Extension 2: illumination-aware conditioning](#7-extension-2-illumination-aware-conditioning)
8. [Experiment protocol](#8-experiment-protocol)
9. [Measurements so far](#9-measurements-so-far)
10. [Problems we hit, and what they teach](#10-problems-we-hit-and-what-they-teach)
11. [Known issues and open questions](#11-known-issues-and-open-questions)
12. [Proposals and future work](#12-proposals-and-future-work)
13. [How to run everything](#13-how-to-run-everything)
14. [Glossary](#14-glossary)
15. [Revision questions](#15-revision-questions)

---

## 1. The problem

Photos taken in very low light are dark, noisy, colour-shifted and lose detail. Simply amplifying the signal amplifies the noise too. The task is **low-light raw enhancement**: given a short-exposure raw sensor image, produce the well-exposed sRGB image a long exposure would have given.

**Our project abstract** sets two goals:

1. **Illumination-aware conditioning.** Estimate an illumination map and use it to steer the model towards regions that need stronger enhancement (Ahon's part).
2. **Lighter backbones.** Replace the large diffusion U-Net with smaller ones, so the method runs on modest hardware (Shamba's part).

Evaluation follows the abstract: PSNR, SSIM, LPIPS for image quality, and parameters, GPU memory, inference time and FLOPs for cost.

---

## 2. Background: raw images and the camera ISP

### 2.1 Raw sensor data

A camera sensor measures one colour per pixel through a **Bayer colour filter array**, usually a repeating 2×2 pattern **RGGB**: one red, two green, one blue. Raw values are linear in light, have a **black level** (the reading for zero light) and a **white level** (saturation).

### 2.2 The ISP

The **image signal processor (ISP)** turns raw data into a viewable sRGB image. The classic steps:

| Step | What it does |
|---|---|
| Black-level subtraction | Removes the sensor offset |
| Normalisation | Scales to [0, 1] using the white level |
| Demosaicing | Rebuilds full RGB from the Bayer mosaic |
| White balance (WB) | Per-channel gains so grey objects look grey |
| Colour correction matrix (CCM) | Maps camera RGB to a standard colour space (sRGB) |
| Denoising | Removes noise, critical in low light |
| Tone mapping / gamma | Non-linear curve for display |

In low light, a short exposure has very few photons, so the signal is tiny and dominated by **shot noise** (Poisson) and **read noise**. Multiplying by the exposure ratio (e.g. ×300) brightens the image but also brightens all that noise.

### 2.3 Why a learned ISP

Hand-tuned ISPs fail in extreme low light. Learned methods (starting with *Learning to See in the Dark*, SID) train a network to map short-exposure raw images straight to long-exposure sRGB images. Regression networks trained with L1/L2 losses produce **blurry, over-smoothed** results, because they average over the many plausible clean images. **Generative models** (like diffusion) can produce sharp, realistic detail instead. That is the motivation for DarkDiff.

---

## 3. Background: diffusion models

### 3.1 Forward and reverse process (DDPM)

A diffusion model learns to reverse a gradual noising process. Given clean data $z_0$, the **forward process** adds Gaussian noise over $T = 1000$ steps. In closed form:

$$z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

$\bar\alpha_t$ falls from about 1 (clean) to about 0 (pure noise). The schedule here is SD's **scaled-linear** $\beta_t$ from 0.00085 to 0.012, with $\bar\alpha_t = \prod_{s \le t}(1-\beta_s)$.

A network $f_\theta(z_t, t, c)$ is trained to undo this, given the noisy sample, the timestep and a condition $c$.

### 3.2 What the network predicts

| Parameterisation | Target | Recover $\hat z_0$ from prediction |
|---|---|---|
| **epsilon** | $\epsilon$ | $\hat z_0 = (z_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ |
| **v-prediction** | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\, z_0$ | $\hat z_0 = \sqrt{\bar\alpha_t}\, z_t - \sqrt{1-\bar\alpha_t}\,\hat v$ |

The loss is plain MSE between prediction and target. SD 2.1 (768 px) uses **v-prediction**. SD 2.1-base, SD 1.x and all the distilled backbones use **epsilon**. Our code handles both (`DarkDiff._target`, `DarkDiff._x0`).

### 3.3 DDIM sampling

**DDIM** is a deterministic sampler that skips steps: 50 steps instead of 1000. Each step predicts $\hat z_0$ and $\hat\epsilon$, then jumps to the next, less noisy, timestep $t'$:

$$z_{t'} = \sqrt{\bar\alpha_{t'}}\, \hat z_0 + \sqrt{1-\bar\alpha_{t'}}\, \hat\epsilon$$

### 3.4 Classifier-free guidance (CFG)

Train one network both **with** and **without** the condition by randomly dropping it. At sampling time, extrapolate away from the unconditional prediction:

$$\hat\epsilon = \hat\epsilon_u + w\,(\hat\epsilon_c - \hat\epsilon_u)$$

$w > 1$ strengthens the condition. DarkDiff uses $w = 2.0$ for SID and LRD, and $2.5$ for ELD.

### 3.5 Latent diffusion and Stable Diffusion

Running diffusion on full-resolution pixels is expensive. **Latent diffusion** first compresses images with a **VAE**: an encoder $E$ maps an image to a latent 8× smaller per side with 4 channels, and a decoder $D$ maps back. Diffusion runs in that latent space. Latents are multiplied by a **scaling factor** (0.18215 for SD) so they have roughly unit variance.

**Stable Diffusion (SD)** is a latent text-to-image diffusion model:
- a VAE (f = 8, 4 latent channels);
- a **U-Net** denoiser;
- a CLIP text encoder whose output the U-Net reads through **cross-attention**.

### 3.6 The SD U-Net

The U-Net has an encoder (down path), a middle block and a decoder (up path), with skip connections between matching resolutions.

| Building block | Contents |
|---|---|
| `ResnetBlock2D` | GroupNorm, SiLU, 3×3 conv, plus the timestep embedding, then GroupNorm, SiLU, 3×3 conv, plus a skip |
| `Transformer2DModel` | Flattens the feature map into tokens and runs a `BasicTransformerBlock` |
| `BasicTransformerBlock` | Self-attention (`attn1`), cross-attention to text (`attn2`) and a feed-forward layer, each with a residual |
| Down / up samplers | Stride-2 conv / nearest upsample + conv |

SD 2.1's U-Net: channels 320/640/1280/1280, 2 layers per down block, 3 per up block, attention head dim 64, text width 1024 (OpenCLIP). About 866M parameters.

### 3.7 Attention, briefly

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right)V$$

$Q$ comes from the image tokens. $K$ and $V$ come from the same tokens (self-attention) or from another source (cross-attention). $B$ is an optional additive bias or mask: $-\infty$ removes a key. Multi-head attention splits channels into $h$ heads of size $d$.

---

## 4. The DarkDiff paper

### 4.1 Core idea

DarkDiff **retasks a pretrained Stable Diffusion model as a camera ISP**. Instead of generating images from text, the diffusion model generates the clean sRGB image, conditioned on the noisy low-light input. Pretraining gives it a strong prior of what natural images look like. The conditioning keeps it faithful to the actual scene.

### 4.2 Pipeline

```
short-exposure raw ──► minimal ISP ──► noisy LRGB (×ratio, white-balanced, linear)
                                          │
                       ┌──────────────────┴──────────────────┐
                       ▼                                     ▼
              VAE encoder E (frozen)                 encoder features E_b
                       │ z_y (condition latent)              │ (skip path)
                       ▼                                     │
              Context processor (U-Net encoder copy)         │
                       │ per-layer context tokens            │
                       ▼                                     │
   noise z_T ──► U-Net denoiser with region cross-attention  │
                 (50 DDIM steps, CFG)                        │
                       │ ẑ_0                                 │
                       ▼                                     ▼
                VAE decoder D  ◄──────── residual skip convs ┘
                       │
                       ▼
                 clean sRGB output
```

### 4.3 Input: noisy linear RGB (LRGB)

Raw data is turned into a simple **linear RGB**: Bayer packing, black-level removal, exposure-ratio gain, white balance, and the two greens averaged. Demosaicing at half resolution avoids interpolation. There's **no colour correction or gamma**: the network learns those.

### 4.4 Residual VAE (paper Sec. 4.3)

The standard VAE bottleneck (8× downsampling, 4 channels) loses fine detail. DarkDiff adds **skip connections from the encoder to the decoder**: each encoder stage's features $E_b$ pass through a conv and are added to the input of the matching decoder stage. So the decoder sees high-resolution information from the actual input, not only the latent.

The VAE is fine-tuned in **stage 1** to map *noisy LRGB in → clean sRGB out*, with the usual latent-diffusion autoencoder loss (Sec. 5.3).

### 4.5 Context processor

The condition latent $z_y = E(\text{noisy})$ is processed by a **copy of the U-Net's encoder** (down blocks + mid block). This yields features at every resolution, with the same shapes as the denoiser's own features, so they can be used directly as attention context.

### 4.6 Region-based cross-attention (paper Eqs. 5–7)

The denoiser's cross-attention (`attn2`) originally attends to text. DarkDiff replaces it in the **down and mid blocks** with attention to the context features. Global attention would let any pixel attend to any other, which is expensive and wrong for an ISP (a pixel's output depends mostly on its neighbourhood). So attention is restricted to **regions**: the token grid is split into $r \times r$ windows, and each window's queries attend only to the context tokens in the *same* window.

### 4.7 Decoder image loss (paper Eqs. 8–9)

The latent loss alone does not guarantee a good *image*. DarkDiff adds an image-space loss. It takes the one-step estimate $\hat z_0$ (from the current prediction), decodes it with the frozen residual decoder, and compares that to the ground-truth sRGB. Gradients flow through the frozen decoder into the U-Net:

$$\mathcal{L} = \mathcal{L}_{LDM} + \lambda\, \mathcal{L}_{image}, \qquad \mathcal{L}_{image} = \big\| D(\hat z_0 / s,\ E_b) - y_{sRGB} \big\|^2$$

### 4.8 CFG with a noise condition (paper Eq. 10)

During training, 5% of condition latents are replaced with pure Gaussian noise. That teaches an unconditional mode. At inference:

$$\hat f = f_u + w\,(f_c - f_u)$$

Here $f_u$ uses a noise condition and $f_c$ the real one.

### 4.9 Training setup (paper Appendix A2)

| | Stage 1 (VAE) | Stage 2 (LDM) |
|---|---|---|
| Optimiser | Adam, lr 4e-5, betas (0.5, 0.9) | Adam: 2.5e-4 for context processor + region attention, 5e-5 for the rest of the U-Net |
| Epochs | 3000 | 3000 |
| Crops | 1200×1200 | 1200×1200 |
| Hardware | 8× A100 | 8× A100 |

Inference: DDIM, 50 steps, guidance 2.0 (SID, LRD) or 2.5 (ELD).

### 4.10 Datasets

| Dataset | Camera | Test protocol |
|---|---|---|
| **SID** (Sony subset) | Sony α7S II | ×300 ratio test pairs |
| **ELD** | Sony α7S II | ×200 subset, evaluated with the **SID-trained** model (cross-dataset generalisation) |
| **LRD** | — | −3EV subset |

---

## 5. Our implementation of DarkDiff

### 5.1 Code map

| File | Role |
|---|---|
| `darkdiff/isp.py` | Minimal ISP: Bayer packing, normalisation, LRGB, CCM, gamma |
| `darkdiff/data.py` | Pair lists for SID / ELD / LRD, `PairDataset` (crops, flips, retrying loader) |
| `darkdiff/vae.py` | `ResidualVAE`, `NLayerDiscriminator`, `VAELoss` |
| `darkdiff/model.py` | `DarkDiff` (condition, context, denoise, losses, sampling), `RegionCrossAttention` |
| `darkdiff/backbones.py` | The pretrained U-Nets and `load_backbone` |
| `darkdiff/illum.py` | Ahon's illumination components |
| `train_vae.py` | Stage 1 |
| `train_ldm.py` | Stage 2 |
| `evaluate.py` | Full-image PSNR / SSIM / LPIPS |
| `sample_images.py` | Side-by-side noisy / output / ground-truth PNGs |
| `benchmark_backbones.py` | Efficiency measurements with random weights |
| `sweep_backbones.py` | Train + evaluate + benchmark + summarise all backbones |
| `download_data.py` | Dataset download and extraction |
| `tests/` | Self-checks with tiny random models; no downloads |

### 5.2 Minimal ISP (`darkdiff/isp.py`)

For each raw file:

1. **Pack** the Bayer mosaic into 4 half-resolution planes R, G1, B, G2 (`pack_bayer`). Some DNG files label both greens as 1, so the second green site is used for G2.
2. **Normalise**: $(x - \text{black}) / (\text{white} - \text{black})$, clamped at 0.
3. **Gain**: multiply the noisy image by the exposure ratio (e.g. ×300).
4. **White balance**: gains from the ground-truth file's `camera_whitebalance`, normalised so G = 1. The same gains go on both images.
5. **LRGB**: $(R,\ (G1+G2)/2,\ B)$, clipped to [0, 1].
6. **sRGB target** (ground truth only):
   - CCM from the camera's `rgb_xyz_matrix`: $M_{cam \leftarrow sRGB} = M_{cam \leftarrow XYZ} \cdot M_{XYZ \leftarrow sRGB}$, rows normalised to sum to 1 (white maps to white), then inverted.
   - Apply the CCM to the clean LRGB, then the sRGB gamma curve.

`load_pair` returns `(noisy_lrgb, clean_lrgb, clean_srgb)`. `PairDataset` maps all three to [−1, 1]. It takes random crops and flips for training, and the full image (cropped to a multiple of 8) for testing.

### 5.3 Datasets (`darkdiff/data.py`)

- **SID**: ratio = long / short exposure, capped at 300, because 10 s / 0.033 s = 303 would otherwise be dropped. Test keeps only ratio-300 pairs: **236 pairs**. Each scene has several noisy shots that share one ground truth.
- **ELD**: SonyA7S2, scenes 1–10, noisy images 5, 10, 15, each with the nearest of ground truths 1, 6, 11, 16. **30 pairs.** The ratio is fixed at the nominal ×200; ELD's per-pair EXIF ratio isn't read.
- **LRD**: −3EV pairs. Ratio taken from the filename exposures.
- Raw reads retry 3 times, 5 s apart, because network drives give transient I/O errors. The final error names both files.

### 5.4 Residual VAE (`darkdiff/vae.py`)

- `ResidualVAE` wraps SD's `AutoencoderKL`.
- `encode` runs the encoder and keeps each down-stage's output $E_b$.
- `decode` adds $\text{Conv}_b(E_b)$ to the input of decoder stage $n-1-b$.
- The skip convs are **zero-initialised**, so before training the model equals the pretrained SD VAE.
- During stage-2 training, the decoder uses **gradient checkpointing per resnet**. This saves about 58% activation memory, because the full-resolution decoder backward otherwise dominated memory (about 28 GB at 1200 px).

**Stage-1 loss** (`VAELoss`, from LDM's `LPIPSWithDiscriminator`):

$$\text{nll} = \frac{|x - \hat x| + \text{LPIPS}_{vgg}(x,\hat x)}{e^{\ell}} + \ell \qquad (\ell \text{ learned log-variance})$$

$$\mathcal{L}_G = \text{nll} + 10^{-6}\,\text{KL} + \lambda_{adapt} \cdot \text{factor} \cdot \big(-D(\hat x)\big)$$

$$\mathcal{L}_D = \tfrac12\big(\text{ReLU}(1 - D(x)) + \text{ReLU}(1 + D(\hat x))\big) \quad \text{(hinge)}$$

- $\lambda_{adapt} = 0.5 \cdot \|\nabla \text{nll}\| / \|\nabla \mathcal{L}_{GAN}\|$, taken at the decoder's last layer and clamped.
- The discriminator is a PatchGAN. It switches on at step 50001.

### 5.5 The DarkDiff model (`darkdiff/model.py`)

**Construction**
- `ctx_net` is a `deepcopy` of the U-Net with `up_blocks`, `conv_norm_out` and `conv_out` deleted. Forward hooks capture the input to every encoder-side `attn2`: these are the context tokens.
- In the main U-Net, every encoder-side `attn2` (down blocks + mid block) is replaced by `RegionCrossAttention`.
- Up blocks keep text cross-attention, fed with the **empty-prompt** CLIP embedding.

**`RegionCrossAttention`**
- Tokens are reshaped to the grid and zero-padded to whole $r \times r$ windows ($r = 8$). Padded keys are masked out.
- Q, K, V and output projections are **initialised from the block's self-attention** (`attn1`). Only shape-matching weights are copied; this matters for SDXS (Sec. 6.3).
- The output projection is **zero-initialised**, so the region branch starts silent and the model starts as the pretrained U-Net.
- It returns no residual, because `BasicTransformerBlock` adds it.

**Forward passes**
- `condition(noisy)`: $z_y = E(\text{noisy}).\text{mode} \cdot s$, plus the encoder features, computed without gradients.
- `context(z_y)`: runs `ctx_net` at **timestep 0** with the empty prompt and returns `{site: tokens}`. Because it doesn't depend on $t$, it runs **once per image**, not once per DDIM step.
- `denoise(z_t, t, ctx, L)`: sets each region layer's context (and illumination map), then runs the U-Net.

**`training_loss`**
1. Compute $z_y$ and features. Encode the target latent $z_0 = E(\text{clean LRGB})$; see the issue in Sec. 11.1.
2. CFG drop: for 5% of samples, replace $z_y$ with Gaussian noise.
3. Sample $t \sim U\{0,\dots,999\}$ and noise; form $z_t$.
4. $\mathcal{L}_{LDM} = \text{MSE}(f(z_t, t, \text{ctx}),\ \text{target})$, where the target is $\epsilon$ or $v$.
5. $\hat z_0$ from the prediction; $\mathcal{L}_{image} = \text{MSE}(D(\hat z_0 / s, E_b),\ y_{sRGB})$.
6. Return $\mathcal{L}_{LDM} + \lambda \mathcal{L}_{image}$, with $\lambda = 1$.

**`sample`**
- Conditional context from $z_y$; unconditional context from Gaussian noise. Both are computed once.
- DDIM for 50 steps from $z_T \sim \mathcal{N}(0, I)$, running both branches as one batch of 2, then CFG.
- Decode with the residual decoder; clamp to [−1, 1].

**Optimiser groups (`param_groups`)**
- lr 2.5e-4 for the context processor + region attention (new or re-purposed parts).
- lr 5e-5 for the rest of the U-Net.
- The VAE is frozen in stage 2.

### 5.6 Training and evaluation scripts

- `train_vae.py` / `train_ldm.py`: bf16 autocast, batch 1 (default), crops of 1200. They save `*_epoch{N}.pt` every `--save-every` epochs (default 100) and always at the end, plus `*_last.pt`. Stage-2 checkpoints hold the U-Net, context processor and region layers, but not the frozen VAE.
- `evaluate.py`: full test images, DDIM 50 steps, PSNR / SSIM on [0, 1], LPIPS with AlexNet. Writes `metrics.json` (mean + per image).
- `--vae-ckpt` is optional. Without it, the pretrained SD VAE is used with zero skips; see Sec. 11.1 for the catch.
- `--gpu N` picks the CUDA device on every script.

### 5.7 Choices the paper does not specify (ours)

| Choice | Our value | Why |
|---|---|---|
| Init of skip convs and region output projection | zero | Start exactly at the pretrained model |
| Init of region Q / K / V | copied from `attn1` | Reuse pretrained attention |
| Region size | 8×8 latent tokens at every layer | Simple, cheap |
| Where region attention goes | down + mid blocks; up blocks keep text attention (empty prompt) | Encoder side reads the condition |
| Context processor timestep | 0 | Makes it time-independent, so it runs once |
| CFG unconditional branch | Gaussian-noise condition | Matches the 5% training drop |
| $\lambda$ | 1.0 | Not given in the paper |
| ELD ratio | fixed ×200 | Avoids parsing EXIF |
| Base model | `sd2-community/stable-diffusion-2-1` | The original `stabilityai` repo is gone. It's the 768-v model (v-prediction) |

---

## 6. Extension 1: lightweight backbones

### 6.1 Goal and constraint

The goal is to cut the U-Net's size and compute while keeping quality. The key constraint is that **DarkDiff fine-tunes a pretrained diffusion prior**, so every backbone must be **pretrained in the SD VAE latent space**. A backbone trained from scratch has no prior, which defeats the method.

### 6.2 Candidates in the project

All numbers below are measured from the configs.

| Backbone | Source | Lineage | U-Net params | Levels | Mid block | Prediction |
|---|---|---|---|---|---|---|
| `sd21` (baseline) | `sd2-community/stable-diffusion-2-1` | SD 2.1-768 | 866M | 4 | yes | v |
| `bk-sdm-v2-base` | `nota-ai/bk-sdm-v2-base` | distilled SD 2.1-base | 584M | 4 | yes | eps |
| `bk-sdm-v2-small` | `nota-ai/bk-sdm-v2-small` | distilled SD 2.1-base | 486M | 4 | no | eps |
| `bk-sdm-v2-tiny` | `nota-ai/bk-sdm-v2-tiny` | distilled SD 2.1-base | 327M | 3 | no | eps |
| `small-sd` | `segmind/small-sd` | distilled SD 1.5 | 582M | 4 | yes | eps |
| `tiny-sd` | `segmind/tiny-sd` | distilled SD 1.5 | 324M | 3 | no | eps |
| `sdxs-512` | `IDKiro/sdxs-512-0.9` | SD 2.1-base, 1-step distilled | 328M | 3 | no | eps |

### 6.3 How the distilled models were made

**BK-SDM (block-removed knowledge distillation).** Starting from SD, whole blocks are removed, and the student is trained to imitate the teacher's outputs and intermediate features:

| Variant | What is removed |
|---|---|
| base | One resnet + transformer pair per stage. Down blocks go from 2 layers to 1; up blocks from 3 to 2 |
| small | base, plus the mid block |
| tiny | small, plus the innermost resolution level (the 1280-channel `DownBlock2D` and its `UpBlock2D`). The deepest feature map becomes 1/32 of the image instead of 1/64 |

The remaining layers keep SD's exact widths, so they're initialised from the teacher's weights before distillation.

Why compute drops less than parameters: most parameters sit in the deep 1280-channel blocks, which run at low resolution. Most FLOPs come from the high-resolution 320- and 640-channel levels, which are kept. For `bk-sdm-v2-tiny`: −62% parameters, but only −40% FLOPs.

**Segmind small-sd / tiny-sd**: the same idea, distilled from SD 1.5. Text width is 768, so they use SD 1.5's CLIP for the empty prompt.

**SDXS-512**: a BK-SDM-tiny-style layout, distilled from SD 2.1-base for **one-step** generation. In its highest-resolution blocks, the self-attention `attn1` is converted to *text* cross-attention (`only_cross_attention: [True, False, False]`). There, our region attention can't copy K/V from `attn1` (shape 1024 → 320), so those keep their default init. It's included as a second ~330M pretrained point with a different training objective.

### 6.4 Why there's no ~150M or MobileNet-style pretrained backbone

We searched for pretrained compact or mobile diffusion U-Nets:
- **SnapFusion, MobileDiffusion, SnapGen**: efficient U-Nets with separable convs and pruned attention. Only papers; **no public weights**.
- **DreamLite Mobile**: 0.39B, a different architecture.
- **OFA small-stable-diffusion**: 579M.

The smallest public SD-compatible U-Nets are about 315–330M: BK-SDM-tiny, tiny-sd and SDXS.

### 6.5 Historical note: our two from-scratch designs (removed)

We first built two small U-Nets of our own. They were removed because they weren't pretrained (Sec. 6.1). The designs are still worth knowing for the report or future work, and the code is in git history (`darkdiff/backbones.py` before commit `a32e354`).

**`compact-unet` (150M, 113 GFLOPs at 512 px, −83%)**: SD 2.1's topology at **half width** (160/320/640/640) with **one layer per block**. Same head counts as SD, so head dim falls to 32. Time embedding 640.

**`mobilenetv3-unet` (100M, 73 GFLOPs, −89%)**: `compact-unet` with every `ResnetBlock2D` replaced by an **MBConv** block:

```
GroupNorm → 1×1 expand (2× out ch) → hardswish
  → 3×3 depthwise + time-embedding → hardswish
  → squeeze-excite (pool → 1×1 → ReLU → 1×1 → hardsigmoid) × features
  → 1×1 project (linear bottleneck) + skip
```

A dense 3×3 conv costs about $9\,C_{in}C_{out}$ per pixel. The depthwise-separable form costs about $2C_{in}C + 9C + 2CC_{out}$.

Departures from MobileNetV3:
- GroupNorm instead of BatchNorm, because batches are tiny.
- Fixed 2× expansion, sized from *output* channels. Input-based expansion made the model bigger than `compact-unet`, because up-block inputs are wide skip concatenations.
- 3×3 kernels only.
- SE in every block.

After 10 epochs from random init, both produced unusable images, which is expected for a diffusion model with no prior.

---

## 7. Extension 2: illumination-aware conditioning

### 7.1 Idea

Dark regions of the input have the lowest signal-to-noise ratio, so the model should treat them differently from bright regions. An **illumination map** $L$ tells every layer how bright each region of the input is. The model can then:
- re-weight the condition features;
- distrust very dark condition tokens;
- decide how much to rely on the condition versus its generative prior.

This "brightening filter in the middle of the model" sits **inside every region cross-attention layer**: after the U-Net encoder features exist, before they attend to the condition.

### 7.2 Components (`darkdiff/illum.py`, wired into `RegionCrossAttention` behind `--illum`)

**Illumination map** (`estimate_illumination`):

$$\ell = 0.2126R + 0.7152G + 0.0722B$$

$$L = \text{smooth}_{3\times3}\Big(\text{clamp}\big(\text{avgpool}_{\to\, h \times w}(\ell),\ 0,\ 1\big)\Big)$$

- It's computed from the gain-amplified noisy LRGB, mapped back to [0, 1].
- Average-pooling to the latent grid (64 pixels per cell) cuts the noise standard deviation by about 8× **before** clamping, so zero-mean noise doesn't bias the estimate upward.
- It's interpolated to each layer's token grid.

**Illumination features**: $\phi(L) = [\,L,\ \log(L + \varepsilon)/\log(1/\varepsilon)\,]$ with $\varepsilon = 10^{-3}$. The log channel turns multiplicative exposure differences into additive ones.

**FiLM on the condition (before K/V)**: `IllumFiLM`, a two-layer conv on $\phi(L)$:

$$c' = c \odot (1 + \gamma(\phi(L))) + \beta(\phi(L))$$

**Per-head key bias**: added to the attention logits, so a query can learn to down-weight keys from very dark, low-SNR tokens:

$$B_{h,k} = b_h \log(L_k + \varepsilon)$$

Padded keys stay at $-\infty$.

**Output gate**: a two-layer conv. It scales the region-attention output per region, between 0× and 2×:

$$g = 1 + \tanh(G(\phi(L))) \in (0, 2)$$

**Zero-initialisation.** The last layers of FiLM and the gate, and $b_h$, all start at zero. At step 0 the illum model is **exactly** vanilla DarkDiff. A test checks this.

**CFG.** When the condition is dropped (5% in training) and in the unconditional sampling branch, $L = 1$. That's neutral, since $\log L = 0$, and carries no information about the input.

**Cost**: about +2M parameters and +2–3% FLOPs. Measured at 256 px: +13–26% train-step time, same memory.

### 7.3 Components present but not wired in

| Component | What it does | Why not used now |
|---|---|---|
| `CurveBrighten` | Zero-DCE-style iterated quadratic curves $a \leftarrow a + \alpha\, a(1-a)$ on the raw input, before the VAE encoder | Training it means backpropagating through the frozen VAE encoder at full resolution (large memory). The input is already gain-amplified and clipped ≥ 0. It would also change the VAE's input distribution |
| `LightContextEncoder` | A small MobileNetV3 replacement for the context processor (latent + illumination in) | Output widths hard-coded for SD 2.1, so it needs adapters per backbone. Random init, whereas the paper's context processor is a pretrained encoder copy. Our context processor already runs once per image, so the saving is parameters and FLOPs only |
| `illum_adaptive_cfg` | Spatially varying guidance, stronger in dark regions: $w = w_b + (w_d - w_b)(1-L)^{\gamma}$ | Weights (3.0 / 1.5) untuned, and they would replace the paper's fixed 2.0 / 2.5 |
| Shifted windows | Alternate layers shift windows by $r/2$ to remove seams | Not needed for the first comparison |

---

## 8. Experiment protocol

### 8.1 What is compared

**Paper DarkDiff** (`sd21`, vanilla) against **our modified DarkDiff** (each lightweight backbone **with** illumination conditioning). `darkdiff/backbones.py` sets `BASELINE = "sd21"` and `variant(b)`. All runs use the **same `--epochs`**.

### 8.2 Pipeline (`sweep_backbones.py`)

For each backbone:
1. **Train**: `train_ldm.py`, with `--illum` for non-baseline backbones. Output in `runs/backbones/<b>` (sd21) or `runs/backbones/<b>-illum`.
2. **Evaluate** on SID and ELD. ELD uses the SID model, as in the paper.
3. **Benchmark**: `benchmark_backbones.py`, random weights.
4. **Summarise**: `runs/backbones/summary.csv`.

Every step is a subprocess, so GPU memory is freed between steps. Steps whose outputs exist are skipped, so the sweep resumes after a crash.

### 8.3 Metrics

| Metric | Meaning | Better |
|---|---|---|
| PSNR (dB) | Pixel fidelity: $10 \log_{10}(1/\text{MSE})$ | higher |
| SSIM (0–1) | Structural similarity: local luminance, contrast and structure | higher |
| LPIPS (0–1) | Learned perceptual distance (deep features) | lower |
| `unet_params_M`, `ctx_params_M`, `trainable_params_M` | Model size | lower |
| `unet_gflops`, `ctx_gflops` | Compute per call | lower |
| `infer_s`, `infer_mem_GB` | Time and peak memory per image (50 steps) | lower |
| `train_step_s`, `train_mem_GB` | Time and peak memory per training step | lower |
| `*_delta` | Row minus `sd21` | positive is better for PSNR / SSIM; negative for LPIPS |
| `pareto` | True if no other row has both lower-or-equal LPIPS and lower-or-equal trainable params, with one strictly lower | — |

The first row of `summary.csv`, `how to read`, repeats these directions. Skip it when loading into pandas.

### 8.4 Visual checks

`sample_images.py --backbone <b> -n N` saves noisy (gamma 2.2 for display) | output | ground truth for the first N **distinct scenes**.

---

## 9. Measurements so far

**A40 benchmark, 1200 px training crops, 1024 px inference, random weights.** These were measured **before** decoder checkpointing and with the old from-scratch models:

| Backbone | Trainable params | U-Net GFLOPs | Infer s | Infer GB | Train s/step | Train GB |
|---|---|---|---|---|---|---|
| sd21 | 1217.7M | 4714 | 8.80 | 11.7 | OOM | OOM |
| bk-sdm-v2-base | 815.1M | 2989 | 5.74 | 9.2 | OOM | OOM |
| bk-sdm-v2-small | 619.1M | 2940 | 5.63 | 8.1 | 2.26 | 40.9 |
| bk-sdm-v2-tiny | 414.3M | 2838 | 5.41 | 7.0 | 2.22 | 38.6 |
| small-sd | 811.0M | 2989 | 6.57 | 9.2 | OOM | OOM |
| tiny-sd | 410.8M | 2838 | 6.23 | 6.9 | 2.29 | 38.5 |

What it showed:
- A ~33 GB training-memory floor for *every* backbone. It came from the full-resolution VAE decoder backward in $\mathcal{L}_{image}$, not from the backbone. Decoder checkpointing fixes most of it (Sec. 10.1).
- Distillation barely reduces per-step compute (2.8k vs 4.7k GFLOPs), because the expensive high-resolution layers are kept.

**First 10-epoch training run** (pretrained SD VAE, no stage 1): SID PSNR about 5.6 dB for `mobilenetv3-unet`. `bk-sdm-v2-tiny` gave blurry, colour-cast but coherent images. `sd21` gave noise-like images (Sec. 11.2).

---

## 10. Problems we hit, and what they teach

### 10.1 Training memory dominated by the decoder
**Symptom**: all backbones need about 33 GB or more per step; three ran out of memory on a 48 GB A40.
**Cause**: $\mathcal{L}_{image}$ backpropagates through the full-resolution VAE decoder (about 28 GB of activations at 1200 px).
**Fix**: gradient checkpointing per decoder resnet. It recomputes activations in the backward pass, trading compute for memory: about −58% activation memory measured.

### 10.2 Stage-1 VAE training too slow
2 h per epoch on an A40 means 3000 epochs is about 250 days. **Workaround**: `--vae-ckpt` is optional, using the pretrained SD VAE with zero skips (no residual path). This keeps the backbone comparison fair, since the VAE is the same everywhere. But see Sec. 11.1.

### 10.3 Latent clipping in the scratch schedulers
The scratch backbones built a `DDPMScheduler` with diffusers defaults, where **`clip_sample=True`**. DDIM then clipped latents to [−1, 1], but SD latents span about ±4, so every output was washed out. Pretrained backbones load SD's config, which has `clip_sample=False`. **Lesson**: always check scheduler configs, not just model weights.

### 10.4 Network-drive I/O errors
`LibRawIOError: Input/output error` on a NAS-mounted file crashed ELD evaluation. Fix: retry reads and report file paths.

### 10.5 Identical sample images
The first 8 SID test pairs are 8 noisy shots of **one** scene. Fix: one pair per ground truth.

### 10.6 SDXS attention shapes
SDXS's highest-resolution `attn1` is a text cross-attention (K/V from 1024-d text). Fix: copy only shape-matching weights into region attention.

---

## 11. Known issues and open questions

### 11.1 Target-latent mismatch without a stage-1 VAE (important, not yet fixed)

`training_loss` uses $z_0 = E(\text{clean LRGB})$ as the diffusion target.

- **With a stage-1 VAE** (trained LRGB → sRGB), $D(E(\text{clean LRGB}))$ is sRGB, so $\mathcal{L}_{LDM}$ and $\mathcal{L}_{image}$ agree.
- **With the plain SD VAE** (`--vae-ckpt` omitted, as in our runs), $D(E(\text{clean LRGB}))$ is **linear camera RGB**, dark and without CCM or gamma. $\mathcal{L}_{image}$, meanwhile, compares against **sRGB**. The two losses push the model towards different images. This likely explains part of the colour cast and poor scores.

**Proposed fix**: without a stage-1 VAE, encode the target as $z_0 = E(\text{sRGB})$, so the pretrained decoder returns sRGB directly.

### 11.2 `sd21` produces noise-like outputs after 10 epochs
The output looks like a latent that was never fully denoised; `bk-sdm-v2-tiny` is blurry but clean. The most likely cause is **training instability**: a 1.22B-parameter full fine-tune at batch 1, Adam with no warmup and no gradient clipping, and image-loss gradients through the decoder. Less likely: an untested v-prediction issue (`sd21` is the only v-prediction backbone) or bf16 attention precision.
**Diagnose**: check the training loss curve. Sample an untrained `sd21` checkpoint: if it's clean, the pipeline is fine and training broke the model.
**Fix**: gradient clipping (norm 1.0) and lr warmup.

### 11.3 The comparison mixes two changes
The baseline is `sd21` vanilla; every other row changes **both** the backbone **and** adds illumination. A difference can't be attributed to either change alone. At least one ablation is needed (Sec. 12).

### 11.4 Short training budget
Paper: 3000 epochs on 8× A100. Us: about 10 epochs on 1 GPU. Absolute scores will be far below the paper's. Conclusions should be **relative**, at equal budget.

### 11.5 Other limitations
- ELD uses a fixed ×200 ratio, not the per-pair EXIF ratio.
- LRD ratios come from filenames and vary widely between scenes. They should be checked against the paper.
- Checkpoints have no optimiser state, so training can't resume.
- `sample_images.py` takes the first N scenes, not a random sample.

---

## 12. Proposals and future work

**Fix first (cheap, high impact)**
1. Target latent from sRGB when no stage-1 VAE is used (Sec. 11.1).
2. Gradient clipping + lr warmup in `train_ldm.py` (Sec. 11.2).

**Make the comparison conclusive**
3. **Ablations**:
   - `bk-sdm-v2-tiny` vanilla vs `bk-sdm-v2-tiny` + illum isolates the illumination effect.
   - `sd21` + illum vs `sd21` vanilla checks it on the big model.
4. Same seeds and the same `--epochs` everywhere. Report mean ± std over test images.

**Efficiency ideas**
5. **LoRA / partial fine-tuning** of the U-Net: far fewer trainable parameters and much less optimiser memory, so `sd21` fits on one GPU.
6. **Fewer sampling steps** (e.g. 20 DDIM steps), or a one-step student. SDXS is already a one-step model.
7. **`LightContextEncoder`** with per-backbone adapters, to shrink the context processor (88–352M today).
8. **Distil our compact / MobileNet designs** from SD 2.1-base, BK-SDM style. That would give a truly pretrained ~100–150M model, at hundreds of GPU-hours.

**Illumination ideas**
9. `illum_adaptive_cfg` with tuned weights: a training-free change, so it's cheap to try.
10. `CurveBrighten` on the condition branch, if memory allows.
11. Shifted region windows to remove seams.

**Quality**
12. Train the stage-1 residual VAE for whatever budget allows (e.g. 1 day at 512 px crops). The GAN term starts only after 50k steps.

---

## 13. How to run everything

```sh
uv sync
uv run download_data.py --root data --datasets sid eld lrd

# optional stage 1 (slow)
uv run train_vae.py --dataset sid --out runs/vae

# efficiency only (no data needed)
uv run benchmark_backbones.py --crop 1200 --size 1024

# full comparison: sd21 vanilla vs lightweight + illum, same epochs
uv run sweep_backbones.py --epochs 10 --backbones sd21 bk-sdm-v2-tiny tiny-sd sdxs-512 \
    --data-root /path/to/data --gpu 0 [--vae-ckpt runs/vae/vae_last.pt]

# visual check
uv run sample_images.py --backbone bk-sdm-v2-tiny --dataset sid -n 8

# tests (tiny random models, no downloads)
uv run tests/test_isp.py; uv run tests/test_vae.py; uv run tests/test_model.py
uv run tests/test_backbones.py; uv run tests/test_benchmark.py; uv run tests/test_smoke.py
```

Use `uv run` / `uv add` only. On a multi-GPU server, pass `--gpu N`. Set `HF_HUB_OFFLINE=1` once models are cached.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| Bayer / RGGB | 2×2 colour-filter mosaic on the sensor |
| Black / white level | Raw value at zero light / saturation |
| LRGB | Linear RGB: packed, black-subtracted, gained, white-balanced raw |
| CCM | Colour correction matrix, camera RGB → sRGB |
| ISP | Image signal processor: raw → display image |
| Exposure ratio | Long / short exposure time (×100, ×250, ×300) |
| VAE | Variational autoencoder; SD's image ↔ latent compressor |
| Latent | 4-channel, 8×-downsampled representation |
| U-Net | Encoder–decoder CNN with skip connections; the denoiser |
| epsilon / v-prediction | What the denoiser outputs: the noise, or the "velocity" mix of noise and data |
| DDPM / DDIM | Original stochastic sampler / deterministic fast sampler |
| CFG | Classifier-free guidance |
| Context processor | Copy of the U-Net encoder that turns the condition into per-layer features |
| Region cross-attention | Window-restricted attention from U-Net tokens to condition tokens |
| FiLM | Feature-wise linear modulation: $x(1+\gamma) + \beta$ |
| SE | Squeeze-and-excitation: channel gating from globally pooled features |
| MBConv | MobileNet inverted residual: expand, depthwise, project |
| Knowledge distillation | Training a small student to mimic a large teacher |
| Gradient checkpointing | Recompute activations in backward to save memory |
| bf16 | 16-bit float with float32's exponent range |
| PSNR / SSIM / LPIPS | Pixel / structural / perceptual quality metrics |
| Pareto-optimal | Not beaten on both quality and size by any other option |

---

## 15. Revision questions

Try answering before checking the referenced section.

1. Why does a regression network produce blurry low-light results, and why does a diffusion model help? (2.3)
2. Write the forward-process equation. What does $\bar\alpha_t$ do as $t$ grows? (3.1)
3. Derive $\hat z_0$ from a v-prediction. (3.2)
4. What does DDIM change compared with DDPM sampling? (3.3)
5. How does DarkDiff obtain an *unconditional* model for CFG? (4.8)
6. Why does DarkDiff add residual skips to the VAE? What is their initial value in our code, and why? (4.4, 5.4)
7. Why restrict cross-attention to regions? What would global attention cost at 1424×2128 px? (4.6)
8. What problem does the decoder image loss solve, and why does it make training memory-heavy? (4.7, 10.1)
9. Why can the context processor run once per image instead of once per DDIM step? (5.5)
10. Why are new layers zero-initialised throughout the project? (5.7, 7.2)
11. Why does `bk-sdm-v2-tiny` cut parameters by 62% but FLOPs by only 40%? (6.3)
12. Why can't we use a backbone trained from scratch with DarkDiff? (6.1)
13. Explain the three illumination mechanisms (FiLM, key bias, gate). Which acts before attention, which during, and which after? (7.2)
14. Why is the illumination map average-pooled *before* clamping? (7.2)
15. Why is $L = 1$ used for the unconditional branch? (7.2)
16. Why does `clip_sample=True` ruin latent-diffusion sampling? (10.3)
17. Explain the target-latent mismatch when no stage-1 VAE is used, and its fix. (11.1)
18. Why can't the current comparison tell whether illumination conditioning helps? What ablation fixes that? (11.3, 12)
19. What does a `pareto = True` row in `summary.csv` mean? (8.3)
20. Name two ways to make `sd21` trainable on a single 48 GB GPU. (10.1, 12)
