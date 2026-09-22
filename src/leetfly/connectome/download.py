"""Download raw connectome and LeetCode data. Every download is skipped if the file already exists."""

import sys
import urllib.request
from pathlib import Path

from leetfly import paths


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def malecns() -> None:
    fetch(f"{paths.MALECNS_BASE}/{paths.MALECNS_ANNOTATIONS.name}", paths.MALECNS_ANNOTATIONS)
    fetch(f"{paths.MALECNS_BASE}/{paths.MALECNS_WEIGHTS.name}", paths.MALECNS_WEIGHTS)


def leetcode() -> None:
    if paths.LEETCODE.exists():
        return
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("newfacade/LeetCodeDataset")
    frames = []
    for split in ds:
        frame = ds[split].to_pandas()
        frame["split"] = split
        frames.append(frame)
    paths.LEETCODE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(paths.LEETCODE)


if __name__ == "__main__":
    malecns()
    leetcode()
