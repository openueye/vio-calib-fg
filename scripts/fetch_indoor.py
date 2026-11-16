"""Download the official TUM VI room1 archive and extract its EuRoC directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import time
import urllib.request

URL = "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "indoor")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "dataset-room1_512_16.tar"
    if not archive.exists():
        partial = archive.with_suffix(".tar.part")
        offset = partial.stat().st_size if partial.exists() else 0
        req = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if offset and response.status != 206:
                raise RuntimeError("Server did not honor resume; keep the partial file and retry later")
            start, received, last = time.monotonic(), offset, offset
            with partial.open("ab" if offset else "wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    received += len(block)
                    if received - last >= 32 * 1024 * 1024:
                        print(f"{received / 2**20:.0f} MiB, {(received-offset)/2**20/(time.monotonic()-start):.1f} MiB/s", flush=True)
                        last = received
        partial.rename(archive)
    target = args.out.resolve()
    with tarfile.open(archive) as tar:
        for member in tar:
            dest = (target / member.name).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f"Archive path outside destination: {member.name}")
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                if dest.exists() and dest.stat().st_size == member.size:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open("wb") as out:
                    while block := src.read(1024 * 1024):
                        out.write(block)
    (target / "download.json").write_text(json.dumps({
        "url": URL, "dataset": "TUM VI room1, 512x512, 16-bit images",
        "source": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
        "archive_bytes": archive.stat().st_size,
    }, indent=2), encoding="utf-8")
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main()
