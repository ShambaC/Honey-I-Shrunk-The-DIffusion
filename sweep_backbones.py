"""Compare paper DarkDiff (sd21, vanilla) with our modified DarkDiff (lightweight backbones + illumination filters).
Every run uses the same --epochs. Each step is a subprocess so GPU memory is freed between runs.
Run: uv run sweep_backbones.py --epochs 20 [--vae-ckpt runs/vae/vae_last.pt]"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from benchmark_backbones import BENCH_COLS, print_table, write_csv
from darkdiff.backbones import BACKBONES, BASELINE, SCRATCH, variant

METRICS = ("psnr", "ssim", "lpips")
HIGHER_BETTER = ("psnr", "ssim")
READ = "how to read"
UNITS = {
    "unet_params_M": "millions of U-Net params", "ctx_params_M": "millions of context-processor params",
    "trainable_params_M": "millions of trained params", "unet_gflops": "GFLOPs per U-Net call",
    "ctx_gflops": "GFLOPs per context-processor call", "infer_s": "seconds per image (all DDIM steps)",
    "infer_mem_GB": "peak GB at inference", "train_step_s": "seconds per training step",
    "train_mem_GB": "peak GB per training step",
}


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


def run_dir(out, backbone):
    # sd21 keeps the plain name; illum runs get a suffix so old vanilla runs are never reused by mistake
    v = variant(backbone)
    return Path(out) / (backbone if v == "vanilla" else f"{backbone}-{v}")


def _bench_rows(out):
    path = Path(out) / "benchmark.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        # CSVs written before the variant column are vanilla
        return {(r["backbone"], r.get("variant") or "vanilla"): r for r in csv.DictReader(f)}


def how_to_read(datasets, cols):
    row = {"backbone": READ, "variant": "vanilla = paper DarkDiff; illum = lightweight backbone + illumination filters"}
    for d in datasets:
        for k in METRICS:
            better = "higher" if k in HIGHER_BETTER else "lower"
            row[f"{d}_{k}"] = f"{better} is better ({d} test set)"
            row[f"{d}_{k}_delta"] = f"this row minus {BASELINE}; {'positive' if better == 'higher' else 'negative'} is better"
    row.update({c: f"lower is better ({UNITS[c]})" for c in BENCH_COLS[2:]})
    row["pareto"] = (f"True: no other row has both lower-or-equal {datasets[0]}_lpips "
                     "and lower-or-equal trainable_params_M, with one strictly lower")
    return {c: row.get(c) for c in cols}


def summarize(out, backbones, datasets):
    """Merge eval means and benchmark rows into <out>/summary.csv, one row per backbone, plus a how-to-read row.
    datasets[0] is the primary: sort and pareto use its LPIPS. Returns the data rows."""
    out = Path(out)
    bench = _bench_rows(out)
    rows = []
    for b in backbones:
        row = {"backbone": b, "variant": variant(b)}
        for d in datasets:
            m = run_dir(out, b) / f"eval_{d}" / "metrics.json"
            mean = json.loads(m.read_text())["mean"] if m.exists() else {}
            row.update({f"{d}_{k}": mean.get(k) for k in METRICS})
        row.update({c: _val(bench.get((b, row["variant"]), {}).get(c)) for c in BENCH_COLS[2:]})
        rows.append(row)

    base = next((r for r in rows if r["backbone"] == BASELINE), None)
    deltas = [f"{d}_{k}_delta" for d in datasets for k in METRICS]
    for r in rows:
        for c in deltas:
            x, y = r[c[:-len("_delta")]], base[c[:-len("_delta")]] if base and r is not base else None
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
    write_csv(out / "summary.csv", [how_to_read(datasets, cols), *rows], cols)
    print_table(rows, cols)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare paper DarkDiff (sd21) with lightweight backbones + illumination filters.")
    p.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES),
                   help=f"{BASELINE} runs vanilla (the baseline every *_delta is measured against); the rest run with --illum")
    p.add_argument("--dataset", default="sid", choices=["sid", "lrd"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--vae-ckpt", help="stage-1 VAE, shared by all backbones; omit to use the pretrained SD VAE")
    p.add_argument("--epochs", type=int, default=3000, help="same for every run, so the comparison is fair")
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
        run = run_dir(out, b)
        illum = ["--illum"] if variant(b) == "illum" else []
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
            print(f"no checkpoint for {b}, skipping eval")
            continue
        for d in datasets:
            if (run / f"eval_{d}" / "metrics.json").exists():
                continue
            subprocess.run(["uv", "run", "evaluate.py", "--backbone", b, "--dataset", d,
                            "--data-root", args.data_root, *vae, *illum, "--ckpt", str(ckpt),
                            "--steps", str(args.steps), "--out", str(run / f"eval_{d}"), "--gpu", str(args.gpu)],
                           check=True)
    have = _bench_rows(out)
    if any((b, variant(b)) not in have for b in args.backbones):
        subprocess.run(["uv", "run", "benchmark_backbones.py", "--backbones", *args.backbones,
                        "--crop", str(args.crop), "--steps", str(args.steps),
                        "--out", str(out / "benchmark.csv"), "--gpu", str(args.gpu)], check=True)
    summarize(out, args.backbones, datasets)


if __name__ == "__main__":
    main()
