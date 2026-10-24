"""End-to-end smoke test of train_vae.py, train_ldm.py, evaluate.py with tiny random models.
No downloads: loaders are patched, and evaluate's LPIPS-alex is swapped for LPIPS-vgg (VGG weights are cached, AlexNet is not).
Run: uv run tests/test_smoke.py
"""
import functools
import glob
import json
import math
import os
import sys
import tempfile
from contextlib import ExitStack
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lpips  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402

import darkdiff.data as data  # noqa: E402
import evaluate  # noqa: E402
import train_ldm  # noqa: E402
import train_vae  # noqa: E402
from darkdiff.vae import ResidualVAE, VAELoss  # noqa: E402
from test_model import DEV, tiny_rvae, tiny_unet  # noqa: E402

PAIRS = [("n0_0.1s.ARW", "g0_10s.ARW", 100.0), ("n1_0.1s.ARW", "g1_10s.ARW", 100.0)]
real_lpips = lpips.LPIPS


def fake_load_pair(noisy_path, gt_path, ratio):
    rng = np.random.default_rng(0)
    return tuple(rng.random((64, 64, 3), dtype=np.float32) for _ in range(3))


def fake_unet_sched(model_id):
    return tiny_unet(), DDPMScheduler(prediction_type="v_prediction")


def fake_prompt_embeds(model_id):
    return torch.randn(1, 77, 32, device=DEV)


def fake_pairs(name, root, split):
    return PAIRS


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as st:
        st.enter_context(mock.patch.object(data, "load_pair", fake_load_pair))
        st.enter_context(mock.patch.object(ResidualVAE, "from_pretrained", lambda model_id: tiny_rvae()))
        st.enter_context(mock.patch.object(train_vae, "VAELoss", functools.partial(VAELoss, disc_start=0)))
        st.enter_context(mock.patch.object(lpips, "LPIPS", lambda net: real_lpips(net="vgg")))
        for mod in (train_ldm, evaluate):
            st.enter_context(mock.patch.object(mod, "load_pretrained", fake_unet_sched))
            st.enter_context(mock.patch.object(mod, "empty_prompt_embeds", fake_prompt_embeds))
        for mod in (train_vae, train_ldm, evaluate):
            st.enter_context(mock.patch.object(mod, "get_pairs", fake_pairs))

        vae_dir, ldm_dir, eval_dir = (os.path.join(tmp, d) for d in ("vae", "ldm", "eval"))
        train_vae.main(["--crop", "64", "--epochs", "1", "--workers", "0", "--out", vae_dir])
        vae_ckpt = os.path.join(vae_dir, "vae_last.pt")
        train_ldm.main(["--vae-ckpt", vae_ckpt, "--crop", "64", "--epochs", "1", "--workers", "0",
                        "--region-size", "4", "--out", ldm_dir])
        ldm_ckpt = os.path.join(ldm_dir, "ldm_last.pt")
        evaluate.main(["--vae-ckpt", vae_ckpt, "--ckpt", ldm_ckpt, "--steps", "2", "--region-size", "4",
                       "--out", eval_dir, "--save-images"])

        assert os.path.exists(os.path.join(vae_dir, "vae_epoch1.pt")) and os.path.exists(vae_ckpt)
        assert os.path.exists(ldm_ckpt)
        sd = torch.load(ldm_ckpt, weights_only=True)
        assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()), "non-finite ldm weights"
        assert len(glob.glob(os.path.join(eval_dir, "*.png"))) == 2
        metrics = json.load(open(os.path.join(eval_dir, "metrics.json")))
        assert len(metrics["per_image"]) == 2
        for k in ("psnr", "ssim", "lpips"):
            assert math.isfinite(metrics["mean"][k]), (k, metrics["mean"])
            assert all(math.isfinite(r[k]) for r in metrics["per_image"]), (k, metrics["per_image"])


if __name__ == "__main__":
    print("device:", DEV)
    test_end_to_end()
    print("ok test_end_to_end")
    print("all passed")
