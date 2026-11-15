"""Copy one SID test pair (noisy + ground-truth raw) into a folder for the Kaggle inference notebook.
Run: uv run export_sample.py --data-root /home/shamba/nas/DarkDiff_Data --out sample"""
import argparse
import json
import shutil
from pathlib import Path

from darkdiff.data import sid_pairs


def main(argv=None):
    p = argparse.ArgumentParser(description="Export one SID x300 test pair and its exposure ratio.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--name", default="10074_00_0.1s", help="stem of the noisy test file; default is the scene in the report")
    p.add_argument("--out", default="sample")
    args = p.parse_args(argv)

    pairs = {Path(n).stem: (n, g, r) for n, g, r in sid_pairs(args.data_root, "test")}
    if args.name not in pairs:
        p.error(f"{args.name} is not a x300 SID test image; e.g. {', '.join(sorted(pairs)[:3])}")
    noisy, gt, ratio = pairs[args.name]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (noisy, gt):
        shutil.copy2(f, out / Path(f).name)
    meta = {"noisy": Path(noisy).name, "gt": Path(gt).name, "ratio": ratio}
    (out / "pair.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta} to {out}")


if __name__ == "__main__":
    main()
