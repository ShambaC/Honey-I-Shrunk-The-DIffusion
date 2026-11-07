"""Train, evaluate and compare every DarkDiff backbone. Each step is a subprocess so GPU memory is freed between runs.
Run: uv run sweep_backbones.py --vae-ckpt runs/vae/vae_last.pt"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from benchmark_backbones import BENCH_COLS, print_table, write_csv
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


def summarize(out, backbones, datasets):
    """Merge eval means and benchmark rows into <out>/summary.csv. datasets[0] is the primary: sort and pareto use its LPIPS."""
    out = Path(out)
    bench = {}
    if (out / "benchmark.csv").exists():
        with open(out / "benchmark.csv", newline="") as f:
            bench = {r["backbone"]: r for r in csv.DictReader(f)}
    rows = []
    for b in backbones:
        row = {"backbone": b}
        for d in datasets:
            m = out / b / f"eval_{d}" / "metrics.json"
            mean = json.loads(m.read_text())["mean"] if m.exists() else {}
            row.update({f"{d}_{k}": mean.get(k) for k in METRICS})
        row.update({c: _val(bench.get(b, {}).get(c)) for c in BENCH_COLS[1:]})
        rows.append(row)

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
    cols = ["backbone", *(f"{d}_{k}" for d in datasets for k in METRICS), *BENCH_COLS[1:], "pareto"]
    write_csv(out / "summary.csv", rows, cols)
    print_table(rows, cols)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Train, evaluate and compare DarkDiff backbones end to end.")
    p.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    p.add_argument("--dataset", default="sid", choices=["sid", "lrd"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--vae-ckpt", required=True, help="stage-1 VAE, shared by all backbones")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--crop", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", default="runs/backbones")
    p.add_argument("--skip-train", action="store_true", help="only evaluate and summarize existing checkpoints")
    args = p.parse_args(argv)

    out = Path(args.out)
    datasets = ["sid", "eld"] if args.dataset == "sid" else ["lrd"]  # paper evaluates ELD with the SID model
    for b in args.backbones:
        run = out / b
        ckpt = run / "ldm" / "ldm_last.pt"
        if not args.skip_train and not ckpt.exists():
            cmd = ["uv", "run", "train_ldm.py", "--backbone", b, "--dataset", args.dataset,
                   "--data-root", args.data_root, "--vae-ckpt", args.vae_ckpt, "--epochs", str(args.epochs),
                   "--crop", str(args.crop), "--batch-size", str(args.batch_size), "--out", str(run / "ldm")]
            if b in SCRATCH:
                cmd += ["--lr-unet", "2.5e-4"]  # random-init U-Nets need the higher lr; 5e-5 is for fine-tuning pretrained
            subprocess.run(cmd, check=True)
        if not ckpt.exists():
            print(f"no checkpoint for {b}, skipping eval")
            continue
        for d in datasets:
            if (run / f"eval_{d}" / "metrics.json").exists():
                continue
            subprocess.run(["uv", "run", "evaluate.py", "--backbone", b, "--dataset", d,
                            "--data-root", args.data_root, "--vae-ckpt", args.vae_ckpt, "--ckpt", str(ckpt),
                            "--steps", str(args.steps), "--out", str(run / f"eval_{d}")], check=True)
    if not (out / "benchmark.csv").exists():
        subprocess.run(["uv", "run", "benchmark_backbones.py", "--backbones", *args.backbones,
                        "--crop", str(args.crop), "--steps", str(args.steps),
                        "--out", str(out / "benchmark.csv")], check=True)
    summarize(out, args.backbones, datasets)


if __name__ == "__main__":
    main()
