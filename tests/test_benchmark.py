"""Self-check for benchmark_backbones (needs CUDA) and sweep_backbones (summarize, commands, skips), with tiny or fake models.
Run: uv run tests/test_benchmark.py"""
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402

import benchmark_backbones as bench  # noqa: E402
import sweep_backbones as sweep  # noqa: E402
from darkdiff.backbones import SCRATCH  # noqa: E402
from test_model import DEV, tiny_rvae, tiny_unet  # noqa: E402

CALLS = []
METRICS_JSON = {"mean": {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2}, "per_image": []}


def fake_backbone(name, weights=True):
    return tiny_unet(), DDPMScheduler(prediction_type="v_prediction"), torch.randn(1, 77, 32, device=DEV)


def fake_run(cmd, check=False):
    """Stands in for subprocess.run: records the command and writes the file the real step would."""
    CALLS.append(cmd)
    if cmd[2] == "train_ldm.py":
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "ldm_last.pt").write_bytes(b"")
    elif cmd[2] == "evaluate.py":
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(METRICS_JSON))
    elif cmd[2] == "benchmark_backbones.py":
        bench.write_csv(cmd[cmd.index("--out") + 1], [bench_row("sd21", 1.0), bench_row("compact-unet", 1.0)],
                        bench.BENCH_COLS)


def bench_row(name, params, train_mem=1.0):
    return {"backbone": name, "unet_params_M": params, "ctx_params_M": 1.0, "trainable_params_M": params,
            "unet_gflops": 1.0, "ctx_gflops": 1.0, "infer_s": 1.0, "infer_mem_GB": 1.0,
            "train_step_s": 1.0, "train_mem_GB": train_mem}


def test_benchmark_csv():
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(bench, "load_backbone", fake_backbone), \
            mock.patch.object(bench, "make_rvae", lambda model_id: tiny_rvae()):
        out = os.path.join(tmp, "benchmark.csv")
        bench.main(["--backbones", "compact-unet", "mobilenetv3-unet", "--size", "64", "--crop", "64",
                    "--steps", "2", "--out", out])
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 2, rows
    for r in rows:
        assert set(r) == set(bench.BENCH_COLS), r.keys()
        for c in bench.BENCH_COLS[1:]:
            assert math.isfinite(float(r[c])), (c, r[c])


def test_summarize_order_and_pareto():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # (lpips, trainable params M). C is dominated by A. D has no eval, so it sorts last with pareto None.
        points = {"A": (0.10, 100.0), "B": (0.20, 50.0), "C": (0.30, 120.0)}
        for name, (lpips, _) in points.items():
            d = out / name / "eval_sid"
            d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps(
                {"mean": {"psnr": 20.0, "ssim": 0.8, "lpips": lpips}, "per_image": []}))
        rows = [bench_row(n, p, "OOM" if n == "B" else 1.0) for n, (_, p) in points.items()]
        rows.append(bench_row("D", 10.0))
        bench.write_csv(out / "benchmark.csv", rows, bench.BENCH_COLS)

        got = sweep.summarize(out, ["C", "A", "B", "D"], ["sid"])
        assert [r["backbone"] for r in got] == ["A", "B", "C", "D"], [r["backbone"] for r in got]
        assert [r["pareto"] for r in got] == [True, True, False, None], [r["pareto"] for r in got]
        assert got[1]["train_mem_GB"] == "OOM", got[1]
        assert (out / "summary.csv").exists()


def test_sweep_commands_and_skips():
    CALLS.clear()
    with tempfile.TemporaryDirectory() as tmp:
        argv = ["--backbones", "sd21", "compact-unet", "--dataset", "sid", "--vae-ckpt", "runs/vae/vae_last.pt",
                "--out", os.path.join(tmp, "runs")]
        with mock.patch.object(sweep.subprocess, "run", fake_run):
            sweep.main(argv)

        trains = [c for c in CALLS if c[2] == "train_ldm.py"]
        assert len(trains) == 2, trains
        for c in trains:
            b = c[c.index("--backbone") + 1]
            assert ("--lr-unet" in c) == (b in SCRATCH), c
            if b in SCRATCH:
                assert c[c.index("--lr-unet") + 1] == "2.5e-4", c
        evals = sorted((c[c.index("--backbone") + 1], c[c.index("--dataset") + 1])
                       for c in CALLS if c[2] == "evaluate.py")
        assert evals == [("compact-unet", "eld"), ("compact-unet", "sid"), ("sd21", "eld"), ("sd21", "sid")], evals

        CALLS.clear()
        with mock.patch.object(sweep.subprocess, "run", fake_run):
            sweep.main(argv)
        assert CALLS == [], CALLS
        assert (Path(tmp) / "runs" / "summary.csv").exists()


if __name__ == "__main__":
    print("device:", DEV)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all passed")
