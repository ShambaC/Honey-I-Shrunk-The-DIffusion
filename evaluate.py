import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm

from darkdiff.data import PairDataset, get_pairs
from darkdiff.model import SD_MODEL, DarkDiff, empty_prompt_embeds, load_pretrained
from darkdiff.vae import ResidualVAE


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample full test images with DarkDiff and report PSNR/SSIM/LPIPS.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="sid", choices=["sid", "eld", "lrd"])
    p.add_argument("--model-id", default=SD_MODEL)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--region-size", type=int, default=8)
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    guidance = args.guidance if args.guidance is not None else (2.5 if args.dataset == "eld" else 2.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    unet, sched = load_pretrained(args.model_id)
    rvae = ResidualVAE.from_pretrained(args.model_id)
    rvae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu", weights_only=True))
    model = DarkDiff(unet, rvae, sched, empty_prompt_embeds(args.model_id), region_size=args.region_size)
    missing, unexpected = model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu", weights_only=True), strict=False)
    # strict=False only so the frozen rvae.* keys (not in the checkpoint) are allowed to be missing
    assert not unexpected and not [k for k in missing if not k.startswith("rvae.")], (missing, unexpected)
    model = model.to(dev).eval()

    lpips_fn = lpips.LPIPS(net="alex").to(dev).eval()
    dl = DataLoader(PairDataset(get_pairs(args.dataset, args.data_root, "test"), None, train=False), batch_size=1)
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    rows = []
    for batch in tqdm(dl, desc="images"):
        with amp:
            pred = model.sample(batch["noisy"].to(dev), steps=args.steps, guidance=guidance)
        pred = (pred.float() + 1) / 2
        gt = (batch["srgb"].to(dev) + 1) / 2
        name = batch["name"][0]
        rows.append({
            "name": name,
            "psnr": peak_signal_noise_ratio(pred, gt, data_range=1.0).item(),
            "ssim": structural_similarity_index_measure(pred, gt, data_range=1.0).item(),
            "lpips": lpips_fn(pred * 2 - 1, gt * 2 - 1).mean().item(),
        })
        if args.save_images:
            save_image(pred, out / f"{name}.png")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    print(f"{args.dataset} n={len(rows)} steps={args.steps} guidance={guidance}: "
          + " ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    (out / "metrics.json").write_text(json.dumps({"mean": mean, "per_image": rows}, indent=2))


if __name__ == "__main__":
    main()
