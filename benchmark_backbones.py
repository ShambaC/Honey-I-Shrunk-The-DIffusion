"""Efficiency of each DarkDiff backbone: params, FLOPs, inference and train-step time and memory.
Random weights and no dataset, since efficiency does not depend on the weights. Run: uv run benchmark_backbones.py"""
import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from darkdiff.backbones import BACKBONES, load_backbone, variant
from darkdiff.model import SD_MODEL, DarkDiff
from darkdiff.vae import ResidualVAE

BENCH_COLS = ["backbone", "variant", "unet_params_M", "ctx_params_M", "trainable_params_M", "unet_gflops", "ctx_gflops",
              "infer_s", "infer_mem_GB", "train_step_s", "train_mem_GB"]


def make_rvae(model_id):
    # Only the VAE JSON config is downloaded; the weights stay random.
    return ResidualVAE(AutoencoderKL.from_config(AutoencoderKL.load_config(model_id, subfolder="vae")))


def gflops(fn):
    # FlopCounterMode counts FLOPs (2 per multiply-add), not MACs.
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e9


def measure(fn, warm):
    """(seconds, peak GB) of fn() after warm(), under bf16 autocast. ("OOM", "OOM") if it does not fit."""
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warm()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**30
    except torch.OutOfMemoryError:
        return "OOM", "OOM"
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def bench(name, args):
    dev = "cuda"
    unet, sched, embeds = load_backbone(name, weights=False)
    model = DarkDiff(unet, make_rvae(args.model_id), sched, embeds, illum=variant(name) == "illum").to(dev)
    groups = model.param_groups(2.5e-4, 5e-5)
    opt = torch.optim.Adam(groups)
    row = {
        "backbone": name,
        "variant": variant(name),
        "unet_params_M": sum(p.numel() for p in model.unet.parameters()) / 1e6,
        "ctx_params_M": sum(p.numel() for p in model.ctx_net.parameters()) / 1e6,
        "trainable_params_M": sum(p.numel() for g in groups for p in g["params"]) / 1e6,
    }

    model.eval()
    s = args.size // 8
    with torch.no_grad():
        z = torch.randn(1, 4, s, s, device=dev)
        t = torch.full((1,), 500, device=dev, dtype=torch.long)
        ctx = model.context(z)
        L = torch.rand(1, 1, s, s, device=dev) if model.illum else None
        row["unet_gflops"] = gflops(lambda: model.denoise(z, t, ctx, L))
        row["ctx_gflops"] = gflops(lambda: model.context(z))

    noisy = torch.rand(1, 3, args.size, args.size, device=dev) * 2 - 1
    row["infer_s"], row["infer_mem_GB"] = measure(
        lambda: model.sample(noisy, steps=args.steps, guidance=args.guidance),
        lambda: model.sample(noisy, steps=2, guidance=args.guidance))

    model.train()
    x = torch.rand(1, 3, args.crop, args.crop, device=dev) * 2 - 1  # one tensor for all three inputs: cost is content-independent

    def train_step():
        opt.zero_grad(set_to_none=True)
        loss, _ = model.training_loss(x, x, x)
        loss.backward()
        opt.step()

    row["train_step_s"], row["train_mem_GB"] = measure(train_step, train_step)
    return row


def write_csv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return "" if v is None else str(v)


def print_table(rows, cols):
    cells = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    for line in [cols, *cells]:
        print("  ".join(v.ljust(w) for v, w in zip(line, widths)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-backbone efficiency: params, GFLOPs, inference and train-step time and memory.")
    p.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    p.add_argument("--model-id", default=SD_MODEL, help="SD VAE config source for ResidualVAE (random weights)")
    p.add_argument("--size", type=int, default=512, help="square inference image side")
    p.add_argument("--crop", type=int, default=512, help="train-step crop side")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--out", default="runs/backbones/benchmark.csv")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index on multi-GPU systems")
    args = p.parse_args(argv)
    torch.cuda.set_device(args.gpu)

    rows = []
    for name in tqdm(args.backbones, desc="backbones"):  # sd21 vanilla, lightweight backbones with --illum
        rows.append(bench(name, args))
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.out, rows, BENCH_COLS)
    print_table(rows, BENCH_COLS)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
