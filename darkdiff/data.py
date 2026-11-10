import random
import re
import time
from pathlib import Path

import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset

from darkdiff.isp import load_pair

_EXPO = re.compile(r"_([\d.]+)s\.\w+$")


def _exposure(path):
    return float(_EXPO.search(path).group(1))


def _to_tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))) * 2 - 1


def _load_retry(noisy_path, gt_path, ratio, tries=3, wait=5.0):
    """load_pair, retried: network mounts (NAS) return transient EIO. The final error names both files."""
    for k in range(tries):
        try:
            return load_pair(noisy_path, gt_path, ratio)
        except (OSError, rawpy.LibRawIOError) as e:
            if k == tries - 1:
                raise OSError(f"cannot read {noisy_path} / {gt_path} after {tries} tries: {e}") from e
            time.sleep(wait)


class PairDataset(Dataset):
    def __init__(self, pairs, crop, train):
        self.pairs = pairs
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        noisy_path, gt_path, ratio = self.pairs[i]
        # shortcut: raw files are decoded on every __getitem__ (no cache), add a cache when loading is the bottleneck
        noisy, clean, srgb = _load_retry(noisy_path, gt_path, ratio)
        H, W = noisy.shape[:2]
        if self.train and self.crop:
            h = w = self.crop
            y, x = random.randint(0, H - h), random.randint(0, W - w)
        else:
            h, w = H // 8 * 8, W // 8 * 8
            y, x = (H - h) // 2, (W - w) // 2
        hflip = self.train and random.random() < 0.5
        vflip = self.train and random.random() < 0.5
        out = []
        for a in (noisy, clean, srgb):
            a = a[y:y + h, x:x + w]
            if hflip:
                a = a[:, ::-1]
            if vflip:
                a = a[::-1]
            out.append(_to_tensor(a))
        return {"noisy": out[0], "clean": out[1], "srgb": out[2], "name": Path(noisy_path).stem}


def sid_pairs(root, split):
    """Sony SID. Lines: `./Sony/short/X_0.1s.ARW ./Sony/long/X_10s.ARW ISO F`."""
    root = Path(root)
    pairs = []
    for line in open(root / f"Sony_{split}_list.txt"):
        short, long_ = line.split()[:2]
        ratio = min(_exposure(long_) / _exposure(short), 300)  # SID caps ratio at 300 (10s/0.033s = 303)
        if split == "test" and ratio != 300:
            continue
        pairs.append((str(root / short), str(root / long_), ratio))
    return pairs


def eld_pairs(root):
    """ELD SonyA7S2, x200 subset: img ids 5,10,15, gt = nearest of 1,6,11,16."""
    # shortcut: ratio fixed at nominal 200, ELD computes ISO*exposure ratio from EXIF per pair (needs exifread)
    d = Path(root) / "SonyA7S2"
    pairs = []
    for scene in range(1, 11):
        for img_id in (5, 10, 15):
            gt_id = min([1, 6, 11, 16], key=lambda g: abs(g - img_id))
            s = d / f"scene-{scene}"
            pairs.append((str(s / f"IMG_{img_id:04d}.ARW"), str(s / f"IMG_{gt_id:04d}.ARW"), 200.0))
    return pairs


def lrd_pairs(root, split):
    """LRD -3EV subset. Lines: `./LRD/Clean/X.DNG ./LRD/Noisy/Y.DNG ISO200 -3EV` (gt first, noisy second)."""
    # shortcut: ratio = filename exposure clean/noisy (same ISO), not the nominal 8x of -3EV; per-scene
    # ratios span ~13-1300x in the train list, so verify the gain against the paper before trusting results
    root = Path(root)
    pairs = []
    for line in open(root / f"LRD_{split}_list.txt"):
        gt, noisy, _, ev = line.split()
        if ev != "-3EV":
            continue
        pairs.append((str(root / noisy), str(root / gt), _exposure(gt) / _exposure(noisy)))
    return pairs


def get_pairs(name, root, split):
    """root is the data dir; datasets live in root/<name>/."""
    root = Path(root) / name
    if name == "sid":
        return sid_pairs(root, split)
    if name == "eld":
        return eld_pairs(root)
    if name == "lrd":
        return lrd_pairs(root, split)
    raise ValueError(f"unknown dataset {name}")
