import argparse
import zipfile
from pathlib import Path

import gdown
import requests
from tqdm import tqdm

SID_ZIP = "https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip"
SID_LISTS = "https://raw.githubusercontent.com/cchen156/Learning-to-See-in-the-Dark/master/dataset/"
ELD_FOLDER_ID = "1QoEhB1P-hNzAc4cRb7RdzyEKktexPVgy"
LRD_ZIP = "https://www.dropbox.com/scl/fi/tari9hvf2ubxo232owekz/LRD.zip?rlkey=7zj9kqgsf06g97smexkrzv05q&dl=1"
LRD_LISTS = "https://raw.githubusercontent.com/fengzhang427/LRD/main/datalist/"

MANUAL = {
    "sid": "Manual: https://drive.google.com/open?id=1G6VruemZtpOyHjOC5N8Ww3ftVXOydSXx (Sony) or Baidu https://pan.baidu.com/s/1fk8EibhBe_M1qG0ax9LQZA (cat SonyPart* > Sony.zip). Put the Sony/ folder in data/sid/.",
    "eld": "Manual: Google Drive folder https://drive.google.com/drive/folders/1QoEhB1P-hNzAc4cRb7RdzyEKktexPVgy or Baidu https://pan.baidu.com/s/11ksugpPH5uyDL-Z6S71Q5g (code 0lby). Put SonyA7S2/ into data/eld/.",
    "lrd": "Manual: Dropbox https://www.dropbox.com/scl/fi/tari9hvf2ubxo232owekz/LRD.zip?rlkey=7zj9kqgsf06g97smexkrzv05q&dl=0 or OneDrive https://1drv.ms/u/s!Ag-MqgbV0Z6NjH3TLuQEF_QszXrR?e=3ugcxy or Baidu https://pan.baidu.com/s/1m7Qwa4Y2IvpAlCfmlgWiUQ (code vj53). Extract so data/lrd/LRD/Clean and data/lrd/LRD/Noisy exist.",
}


def http_download(url, dest):
    """Resumable download; skips if dest already has the full Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"skip {dest.name}, already complete")
        return
    headers, mode = {}, "wb"
    if total and 0 < have < total:
        headers["Range"], mode = f"bytes={have}-", "ab"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"{url} returned an HTML page, not a file")
        if r.status_code == 200:  # server ignored Range
            mode, have = "wb", 0
        with dest.open(mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_path, out_dir):
    with zipfile.ZipFile(zip_path) as z:
        for info in tqdm(z.infolist(), desc=f"extract {zip_path.name}"):
            target = out_dir / info.filename
            if info.is_dir() or (target.exists() and target.stat().st_size == info.file_size):
                continue
            z.extract(info, out_dir)


def get_sid(root):
    d = root / "sid"
    http_download(SID_ZIP, d / "Sony2025.zip")
    extract(d / "Sony2025.zip", d)
    for name in ("Sony_train_list.txt", "Sony_test_list.txt"):
        http_download(SID_LISTS + name, d / name)


def get_eld(root):
    d = root / "eld"
    gdown.download_folder(id=ELD_FOLDER_ID, output=str(d), quiet=False, remaining_ok=True)
    for z in d.glob("*.zip"):
        extract(z, d)
    if not (d / "SonyA7S2").exists():
        raise RuntimeError("SonyA7S2/ not found in data/eld after download")


def get_lrd(root):
    d = root / "lrd"
    http_download(LRD_ZIP, d / "LRD.zip")
    extract(d / "LRD.zip", d)
    for split in ("train", "test"):
        http_download(f"{LRD_LISTS}LRD_{split}_list.txt", d / f"LRD_{split}_list.txt")


GETTERS = {"sid": get_sid, "eld": get_eld, "lrd": get_lrd}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--datasets", nargs="+", default=["sid", "eld", "lrd"], choices=list(GETTERS))
    args = ap.parse_args()
    root = Path(args.root)
    for name in args.datasets:
        print(f"== {name}")
        try:
            GETTERS[name](root)
        except Exception as e:  # keep going so the other datasets still download
            print(f"failed: {e}\n{MANUAL[name]}")
