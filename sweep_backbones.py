"""Train, evaluate and compare every DarkDiff backbone, vanilla and with the illumination-aware cross-attention.
Each step is a subprocess so GPU memory is freed between runs.
Run: uv run sweep_backbones.py --epochs 20 [--vae-ckpt runs/vae/vae_last.pt]"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from benchmark_backbones import BENCH_COLS, VARIANTS, print_table, write_csv
from darkdiff.backbones import BACKBONES, SCRATCH

METRICS = ("psnr", "ssim", "lpips")


def _val(v):
    """CSV cell -> float, the raw text (e.g. "OOM") or None when empty."""
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def _num(v):
    return isinstance(v, float)


def run_dir(out, backbone, variant):
    # vanilla keeps the pre-illum layout, so existing runs/backbones/<backbone> checkpoints are reused
    return Path(out) / (backbone if variant == "vanilla" else f"{backbone}-{variant}")


def _bench_rows(out):
    path = Path(out) / "benchmark.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        # CSVs written before the variant column are vanilla
        return {(r["backbone"], r.get("variant") or "vanilla"): r for r in csv.DictReader(f)}


def summarize(out, backbones, datasets, variants=("vanilla",)):
    """Merge eval means and benchmark rows into <out>/summary.csv, one row per (backbone, variant).
    datasets[0] is the primary: sort and pareto use its LPIPS. *_delta columns on non-vanilla rows are
    (variant - vanilla) for the same backbone: positive is better for psnr/ssim, negative for lpips."""
    out = Path(out)
    bench = _bench_rows(out)
    rows = []
    for b in backbones:
        for v in variants:
            row = {"backbone": b, "variant": v}
            for d in datasets:
                m = run_dir(out, b, v) / f"eval_{d}" / "metrics.json"
                mean = json.loads(m.read_text())["mean"] if m.exists() else {}
                row.update({f"{d}_{k}": mean.get(k) for k in METRICS})
            row.update({c: _val(bench.get((b, v), {}).get(c)) for c in BENCH_COLS[2:]})
            rows.append(row)

    base = {r["backbone"]: r for r in rows if r["variant"] == "vanilla"}
    deltas = [f"{d}_{k}_delta" for d in datasets for k in METRICS]
    for r in rows:
        ref = base.get(r["backbone"]) if r["variant"] != "vanilla" else None
        for c in deltas:
            x, y = r[c[:-6]], ref[c[:-6]] if ref else None
            r[c] = x - y if _num(x) and _num(y) else None

    key = f"{datasets[0]}_lpips"
    rows.sort(key=lambda r: (not _num(r[key]), r[key] if _num(r[key]) else 0.0))
    for r in rows:
        x, y = r[key], r["trainable_params_M"]
        r["pareto"] = None
        if _num(x) and _num(y):
            r["pareto"] = not any(_num(o[key]) and _num(o["trainable_params_M"])
                                  and o[key] <= x and o["trainable_params_M"] <= y
                                  and (o[key] < x or o["trainable_params_M"] < y)
                                  for o in rows if o is not r)
    cols = ["backbone", "variant", *(f"{d}_{k}" for d in datasets for k in METRICS), *deltas,
            *BENCH_COLS[2:], "pareto"]
    write_csv(out / "summary.csv", rows, cols)
    print_table(rows, cols)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Train, evaluate and compare DarkDiff backbones end to end.")
    p.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                   help="vanilla DarkDiff and/or the illumination-aware region cross-attention (--illum)")
    p.add_argument("--dataset", default="sid", choices=["sid", "lrd"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--vae-ckpt", help="stage-1 VAE, shared by all backbones; omit to use the pretrained SD VAE")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--crop", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", default="runs/backbones")
    p.add_argument("--skip-train", action="store_true", help="only evaluate and summarize existing checkpoints")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index on multi-GPU systems")
    args = p.parse_args(argv)

    out = Path(args.out)
    datasets = ["sid", "eld"] if args.dataset == "sid" else ["lrd"]  # paper evaluates ELD with the SID model
    vae = ["--vae-ckpt", args.vae_ckpt] if args.vae_ckpt else []
    for b in args.backbones:
        for v in args.variants:
            run = run_dir(out, b, v)
            illum = ["--illum"] if v == "illum" else []
            ckpt = run / "ldm" / "ldm_last.pt"
            if not args.skip_train and not ckpt.exists():
                cmd = ["uv", "run", "train_ldm.py", "--backbone", b, "--dataset", args.dataset,
                       "--data-root", args.data_root, *vae, *illum, "--epochs", str(args.epochs),
                       "--crop", str(args.crop), "--batch-size", str(args.batch_size), "--out", str(run / "ldm"),
                       "--gpu", str(args.gpu)]
                if b in SCRATCH:
                    cmd += ["--lr-unet", "2.5e-4"]  # random-init U-Nets need the higher lr; 5e-5 is for fine-tuning pretrained
                subprocess.run(cmd, check=True)
            if not ckpt.exists():
                print(f"no checkpoint for {b} ({v}), skipping eval")
                continue
            for d in datasets:
                if (run / f"eval_{d}" / "metrics.json").exists():
                    continue
                subprocess.run(["uv", "run", "evaluate.py", "--backbone", b, "--dataset", d,
                                "--data-root", args.data_root, *vae, *illum, "--ckpt", str(ckpt),
                                "--steps", str(args.steps), "--out", str(run / f"eval_{d}"), "--gpu", str(args.gpu)],
                               check=True)
    have = _bench_rows(out)
    if any((b, v) not in have for b in args.backbones for v in args.variants):
        subprocess.run(["uv", "run", "benchmark_backbones.py", "--backbones", *args.backbones,
                        "--variants", *args.variants, "--crop", str(args.crop), "--steps", str(args.steps),
                        "--out", str(out / "benchmark.csv"), "--gpu", str(args.gpu)], check=True)
    summarize(out, args.backbones, datasets, args.variants)


if __name__ == "__main__":
    main()
