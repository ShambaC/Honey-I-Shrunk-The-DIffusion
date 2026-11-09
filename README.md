# DarkDiff

Unofficial implementation of *DarkDiff: Advancing Low-Light Raw Enhancement by Retasking Diffusion Models for Camera ISP* (arXiv 2505.23743).

## Layout

| File | Paper |
|---|---|
| `darkdiff/isp.py` | Sec 4.1 / A2: Bayer pack, black/white level, digital gain, WB, green binning -> LRGB; CCM + gamma -> sRGB target |
| `darkdiff/data.py` | SID (Sony, x300 test), ELD (SonyA7S2, x200), LRD (-3EV) pair lists + `PairDataset` |
| `darkdiff/vae.py` | Sec 4.3: residual VAE (Fig 7, Tab A1) + LDM stage-1 loss (L1 + LPIPS + PatchGAN + KL) |
| `darkdiff/model.py` | Sec 4.2/4.4: context processor, region-based cross-attention, decoder image loss, CFG sampling (Eq 10) |
| `train_vae.py` | Stage 1: noisy LRGB -> clean sRGB VAE fine-tune |
| `train_ldm.py` | Stage 2: U-Net + context processor + region cross-attention, `L_LDM + lam * L_image` |
| `evaluate.py` | DDIM 50 steps, PSNR / SSIM / LPIPS, full images |
| `download_data.py` | SID / ELD / LRD download + extract |

Base model: `sd2-community/stable-diffusion-2-1` (mirror of SD 2.1; the original `stabilityai` repo is gone). It is v-prediction; the code handles both `epsilon` and `v_prediction`.

## Usage

```sh
uv sync
uv run download_data.py --root data --datasets sid eld lrd
uv run train_vae.py --dataset sid --out runs/vae
uv run train_ldm.py --dataset sid --vae-ckpt runs/vae/vae_last.pt --out runs/ldm
uv run evaluate.py --dataset sid --vae-ckpt runs/vae/vae_last.pt --ckpt runs/ldm/ldm_last.pt --save-images
uv run evaluate.py --dataset eld --vae-ckpt runs/vae/vae_last.pt --ckpt runs/ldm/ldm_last.pt  # SID model, guidance 2.5
```

## Lightweight backbones

`--backbone` on `train_ldm.py` / `evaluate.py` picks the U-Net (`darkdiff/backbones.py`):

| Name | Source | Notes |
|---|---|---|
| `sd21` | `sd2-community/stable-diffusion-2-1` | Full SD 2.1 U-Net, the paper baseline |
| `bk-sdm-v2-base`, `-small`, `-tiny` | `nota-ai/bk-sdm-v2-*` | Pretrained, distilled from SD 2.1-base, blocks removed |
| `small-sd`, `tiny-sd` | `segmind/small-sd`, `segmind/tiny-sd` | Pretrained, distilled from SD 1.5 |
| `compact-unet` | from scratch | SD 2.1 layout at half width, one layer per block |
| `mobilenetv3-unet` | from scratch | `compact-unet` with MobileNetV3-style inverted residual blocks |

```sh
uv run train_vae.py --dataset sid --out runs/vae                     # stage 1, once; shared by all backbones
uv run benchmark_backbones.py                                         # efficiency only: no data, no checkpoints
uv run sweep_backbones.py --vae-ckpt runs/vae/vae_last.pt             # train + evaluate all backbones, then summarize
uv run sweep_backbones.py                                             # same, pretrained SD VAE: skips stage 1, no residual path
```

`sweep_backbones.py` writes `runs/backbones/summary.csv`, one row per backbone: PSNR / SSIM / LPIPS for `sid` and `eld` (or `lrd`), the benchmark columns (`unet_params_M`, `ctx_params_M`, `trainable_params_M`, `unet_gflops`, `ctx_gflops`, `infer_s`, `infer_mem_GB`, `train_step_s`, `train_mem_GB`), and `pareto`. Rows are sorted by primary-dataset LPIPS. `pareto` is True when no other backbone has both lower-or-equal LPIPS and lower-or-equal `trainable_params_M`, with one strictly lower. `OOM` in the benchmark marks a run that does not fit the GPU.

Tests (tiny random models, no downloads): `uv run tests/test_isp.py`, `uv run tests/test_vae.py`, `uv run tests/test_model.py`, `uv run tests/test_backbones.py`, `uv run tests/test_benchmark.py`, `uv run tests/test_smoke.py`.

## Choices the paper does not specify

- Residual VAE skip convs and region cross-attention output projections are zero-initialised, so training starts from the pretrained weights. Region attention Q/K/V are initialised from the block's self-attention.
- Region size is 8x8 latent tokens at every layer (`--region-size`). Region cross-attention replaces text cross-attention in the U-Net down and mid blocks. The up blocks keep text cross-attention with the empty prompt.
- The context processor (a copy of the U-Net encoder) runs at timestep 0. It is computed once per image and reused at every sampling step.
- For CFG, the unconditional branch uses a Gaussian-noise condition latent, which matches the 5% drop used in training.
- `lam` defaults to 1.0.
- ELD uses a fixed x200 ratio rather than the EXIF-derived one. LRD uses the per-pair exposure ratio from the filenames.

Defaults follow paper A2 (1200 crops, 3000 epochs, 8x A100). The full SD 2.1 U-Net, its encoder copy and Adam state do not fit in 12 GB without changes such as 8-bit Adam or gradient checkpointing.
