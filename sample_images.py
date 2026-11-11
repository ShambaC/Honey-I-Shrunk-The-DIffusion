"""Save N side-by-side comparisons (noisy input | DarkDiff output | ground truth) from a trained checkpoint.
Run: uv run sample_images.py --backbone bk-sdm-v2-tiny --dataset sid -n 4"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchvision.utils import save_image
from tqdm import tqdm

from darkdiff.backbones import BACKBONES
from darkdiff.data import PairDataset, get_pairs
from darkdiff.model import SD_MODEL
from evaluate import load_model
from sweep_backbones import run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description="Save noisy | output | ground-truth comparisons for N test images.")
    p.add_argument("--backbone", required=True, choices=BACKBONES)
    p.add_argument("--ckpt", help="stage-2 checkpoint; default is where sweep_backbones.py saves this backbone, "
                                  "runs/backbones/<backbone>[-illum]/ldm/ldm_last.pt")
    p.add_argument("--dataset", default="sid", choices=["sid", "eld", "lrd"])
    p.add_argument("--data-root", default="data")
    p.add_argument("-n", "--num", type=int, default=4, help="number of test scenes (first N, one noisy shot per scene)")
    p.add_argument("--vae-ckpt", help="stage-1 VAE the checkpoint was trained with; omit for the pretrained SD VAE")
    p.add_argument("--model-id", default=SD_MODEL)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--out", help="default runs/samples/<backbone>_<dataset>")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index on multi-GPU systems")
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(args.gpu)
    ckpt = Path(args.ckpt or run_dir("runs/backbones", args.backbone) / "ldm" / "ldm_last.pt")
    if not ckpt.exists():
        p.error(f"no checkpoint at {ckpt}; train it first or pass --ckpt")
    out = Path(args.out or f"runs/samples/{args.backbone}_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    guidance = args.guidance if args.guidance is not None else (2.5 if args.dataset == "eld" else 2.0)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    illum = any(".film." in k for k in state)  # read from the checkpoint, so it always matches how it was trained
    model = load_model(args.backbone, state, args.vae_ckpt, illum, args.region_size, args.model_id).to(dev).eval()
    print(f"{args.backbone} ({'illum' if illum else 'vanilla'}) from {ckpt}")

    pairs = get_pairs(args.dataset, args.data_root, "test")
    first = {}  # SID has several noisy shots per scene sharing one ground truth: keep one per ground truth
    for i, (_, gt, _) in enumerate(pairs):
        first.setdefault(gt, i)
    ds = PairDataset(pairs, None, train=False)
    dl = DataLoader(Subset(ds, list(first.values())[:args.num]), batch_size=1)
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    for batch in tqdm(dl, desc="samples"):
        with amp:
            pred = model.sample(batch["noisy"].to(dev), steps=args.steps, guidance=guidance)
        pred = (pred.float() + 1) / 2
        gt = (batch["srgb"].to(dev) + 1) / 2
        # input is linear RGB: gamma 2.2 for display only, otherwise it looks almost black
        noisy = ((batch["noisy"].to(dev) + 1) / 2).clamp(0, 1) ** (1 / 2.2)
        name = batch["name"][0]
        save_image(torch.cat([noisy, pred, gt], dim=3), out / f"{name}.png")
        print(f"{name}: psnr={peak_signal_noise_ratio(pred, gt, data_range=1.0).item():.2f} dB")
    print(f"wrote {len(dl)} comparisons (noisy | output | ground truth) to {out}")


if __name__ == "__main__":
    main()
