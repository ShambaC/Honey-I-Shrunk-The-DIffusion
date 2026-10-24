import numpy as np
import rawpy

# D65 sRGB -> XYZ (inverse of the usual XYZ -> sRGB matrix)
XYZ_FROM_SRGB = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def pack_bayer(raw, pattern):
    """raw: HxW visible Bayer, pattern: 2x2 color idx (0=R,1=G,2=B,3=G2) -> 4 x H/2 x W/2 planes (R,G1,B,G2)."""
    planes = []
    for color in (0, 1, 2, 3):
        # some files (e.g. DNG) label both greens 1: take the second G site for G2
        pos = np.argwhere(pattern == color) if color != 3 or (pattern == 3).any() else np.argwhere(pattern == 1)[1:]
        dy, dx = pos[0]
        planes.append(raw[dy::2, dx::2])
    return np.stack(planes)


def normalize(planes, black, white):
    return np.maximum((planes - black) / (white - black), 0)


def to_lrgb(planes, wb):
    """planes: 4 x H x W (R,G1,B,G2). wb: 3 gains with G=1. Returns HxWx3 in [0,1]."""
    rgb = np.stack([planes[0], (planes[1] + planes[3]) / 2, planes[2]], -1)
    return np.clip(rgb * wb, 0, 1).astype(np.float32)


def srgb_from_cam(rgb_xyz):
    """rgb_xyz: 3x3 (rawpy rgb_xyz_matrix[:3]) -> 3x3 matrix taking white-balanced camera RGB to linear sRGB."""
    m = rgb_xyz @ XYZ_FROM_SRGB  # cam_from_srgb
    m = m / m.sum(axis=1, keepdims=True)
    return np.linalg.inv(m)


def srgb_gamma(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def _planes(raw, ratio=1.0):
    x = raw.raw_image_visible.astype(np.float32)
    planes = pack_bayer(x, raw.raw_colors_visible[:2, :2])
    black = np.asarray(raw.black_level_per_channel, np.float32)[:, None, None]
    return normalize(planes, black, raw.white_level) * ratio


def load_pair(noisy_path, gt_path, ratio):
    """Returns (noisy_lrgb, clean_lrgb, clean_srgb), each HxWx3 float32 in [0,1], at half resolution."""
    # shortcut: decodes both raw files on every call, no cache, upgrade if loading is the bottleneck
    with rawpy.imread(noisy_path) as n, rawpy.imread(gt_path) as g:
        wb = np.asarray(g.camera_whitebalance[:3], np.float64)
        wb = wb / wb[1]
        noisy = to_lrgb(_planes(n, ratio), wb)
        clean = to_lrgb(_planes(g), wb)
        srgb = srgb_gamma(clean @ srgb_from_cam(g.rgb_xyz_matrix[:3]).T)
    return noisy, clean, srgb
