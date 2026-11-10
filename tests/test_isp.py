import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import darkdiff.data as data  # noqa: E402
from darkdiff.isp import normalize, pack_bayer, srgb_from_cam, srgb_gamma, to_lrgb, XYZ_FROM_SRGB  # noqa: E402

PATTERN = np.array([[0, 1], [3, 2]])  # RGGB, 3 = G2


def test_pack_bayer():
    raw = np.arange(64, dtype=np.float32).reshape(8, 8)
    p = pack_bayer(raw, PATTERN)
    assert p.shape == (4, 4, 4)
    assert np.array_equal(p[0], raw[0::2, 0::2])  # R
    assert np.array_equal(p[1], raw[0::2, 1::2])  # G1
    assert np.array_equal(p[2], raw[1::2, 1::2])  # B
    assert np.array_equal(p[3], raw[1::2, 0::2])  # G2
    # both greens labelled 1 (some DNGs) must give the same planes
    assert np.array_equal(pack_bayer(raw, np.where(PATTERN == 3, 1, PATTERN)), p)


def test_normalize_and_lrgb():
    planes = np.stack([np.full((2, 2), v, np.float32) for v in (6, 6, 6, 6)])
    n = normalize(planes, 2.0, 10.0)
    assert np.allclose(n, 0.5)
    assert np.allclose(normalize(np.full((1, 1), 1.0), 2.0, 10.0), 0)  # clipped below black
    rgb = to_lrgb(np.stack([np.full((2, 2), v, np.float32) for v in (0.25, 0.5, 0.25, 0.5)]), np.array([2.0, 1.0, 1.0]))
    assert rgb.shape == (2, 2, 3)
    assert np.allclose(rgb[..., 0], 0.5) and np.allclose(rgb[..., 1], 0.5) and np.allclose(rgb[..., 2], 0.25)


def test_gamma():
    x = np.array([0.0, 1.0, 0.5, -0.2, 1.3])
    g = srgb_gamma(x)
    assert np.isclose(g[0], 0) and np.isclose(g[1], 1)
    assert np.isclose(g[2], 0.7354, atol=1e-3)
    assert g[3] == 0 and g[4] == 1


def test_color_matrix_identity():
    # camera == sRGB -> correction must be identity
    cam_from_xyz = np.linalg.inv(XYZ_FROM_SRGB) * 3.0  # scale must not matter after row normalisation
    assert np.allclose(srgb_from_cam(cam_from_xyz), np.eye(3), atol=1e-6)


class FakeLoad:
    """Returns a deterministic HxWx3 image per call, so crop/flip consistency can be checked."""

    def __init__(self, H, W):
        self.H, self.W = H, W

    def __call__(self, noisy, gt, ratio):
        rng = np.random.default_rng(0)
        a = rng.random((self.H, self.W, 3)).astype(np.float32)
        return a, a.copy(), a.copy()


def test_dataset_shapes_and_consistency():
    real = data.load_pair
    data.load_pair = FakeLoad(37, 53)
    try:
        pairs = [("a.ARW", "b.ARW", 100.0)]
        ev = data.PairDataset(pairs, crop=None, train=False)[0]
        assert ev["noisy"].shape == (3, 32, 48), ev["noisy"].shape  # center crop to multiple of 8
        assert ev["noisy"].min() >= -1 and ev["noisy"].max() <= 1
        assert ev["name"] == "a"

        tr = data.PairDataset(pairs * 20, crop=16, train=True)
        for i in range(20):
            s = tr[i]
            assert s["noisy"].shape == s["clean"].shape == s["srgb"].shape == (3, 16, 16)
            assert (s["noisy"] == s["clean"]).all()  # same crop and flips for all three
    finally:
        data.load_pair = real


def test_load_retry():
    import rawpy
    from unittest import mock
    calls = []

    def flaky(n, g, r):
        calls.append(n)
        if len(calls) == 1:
            raise rawpy.LibRawIOError(b"Input/output error")
        return "ok"

    with mock.patch.object(data, "load_pair", flaky):
        assert data._load_retry("n.ARW", "g.ARW", 1.0, wait=0) == "ok" and len(calls) == 2
    with mock.patch.object(data, "load_pair", mock.Mock(side_effect=OSError("EIO"))):
        try:
            data._load_retry("n.ARW", "g.ARW", 1.0, wait=0)
            raise AssertionError("expected OSError")
        except OSError as e:
            assert "n.ARW" in str(e) and "g.ARW" in str(e), e


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all passed")
