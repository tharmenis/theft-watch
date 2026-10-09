#!/usr/bin/env python3
"""
Theft Watch - Phase 3, step 1: inventory the raw video dataset.

Walks a folder (default: data/raw), opens every video with PyAV, and records
codec, resolution, fps, duration, whether it decodes, and a checksum for
duplicate detection. Writes one row per file to data/interim/inventory.csv
and prints a summary to paste back to Claude.

Run inside the trainer container, from the repo root:
    docker compose exec trainer python training/inspect_dataset.py data/raw

Options:
    --out PATH     where to write the CSV (default data/interim/inventory.csv)
    --no-hash      skip checksums (faster on very large packs; no duplicate check)
"""
import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import av
except ImportError:
    sys.exit("PyAV is missing. Run this inside the trainer container (it has 'av').")

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".wmv",
             ".m4v", ".webm", ".ts", ".flv", ".3gp"}
# Guess the original video a clip was cut from, e.g. Shoplifting001_x264_12 -> Shoplifting001_x264.
# Only a heuristic: check the "source_id" column before using it for splits.
SOURCE_RE = re.compile(r"^(.*?)(?:[_-]\d{1,4})?$")


def sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> dict:
    info = {"ok": False, "error": "", "codec": "", "width": "", "height": "",
            "fps": "", "duration_s": "", "frames": ""}
    try:
        with av.open(str(path)) as container:
            vs = next((s for s in container.streams if s.type == "video"), None)
            if vs is None:
                info["error"] = "no video stream"
                return info
            cc = vs.codec_context
            info["codec"] = cc.name
            info["width"], info["height"] = cc.width, cc.height
            rate = vs.average_rate or vs.base_rate
            info["fps"] = round(float(rate), 2) if rate else ""
            dur = None
            if vs.duration and vs.time_base:
                dur = float(vs.duration * vs.time_base)
            elif container.duration:
                dur = container.duration / 1_000_000
            info["duration_s"] = round(dur, 2) if dur else ""
            info["frames"] = vs.frames or ""
            # Decode a few frames to prove the file is actually readable.
            decoded = 0
            for _ in container.decode(video=0):
                decoded += 1
                if decoded >= 5:
                    break
            if decoded == 0:
                info["error"] = "no decodable frames"
                return info
            info["ok"] = True
    except Exception as e:  # broken / truncated / unsupported files
        info["error"] = f"{type(e).__name__}: {e}"[:200]
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="data/raw")
    ap.add_argument("--out", default="data/interim/inventory.csv")
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    others = [p for p in files if p.suffix.lower() not in VIDEO_EXT]
    if not videos:
        sys.exit(f"No video files under {root}. Extensions looked for: {sorted(VIDEO_EXT)}")

    rows = []
    for i, p in enumerate(videos, 1):
        rel = p.relative_to(root)
        parts = rel.parts
        row = {
            "path": str(rel),
            "pack": parts[0] if len(parts) > 1 else "(root)",
            "folder": str(rel.parent),
            "file": p.name,
            "source_id": SOURCE_RE.match(p.stem).group(1),
            "size_mb": round(p.stat().st_size / 1_048_576, 2),
        }
        row.update(probe(p))
        row["sha1"] = "" if args.no_hash else sha1(p)
        rows.append(row)
        if i % 50 == 0 or i == len(videos):
            print(f"  probed {i}/{len(videos)}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- summary ----------------
    print("\n=== THEFT WATCH DATASET INVENTORY ===")
    print(f"root: {root}   videos: {len(videos)}   other files: {len(others)}")
    ok = [r for r in rows if r["ok"]]
    print(f"readable: {len(ok)}   broken: {len(rows) - len(ok)}")
    total_min = sum(float(r["duration_s"] or 0) for r in ok) / 60
    print(f"total duration (readable): {total_min:.1f} min")

    print("\n-- per folder (folder names often = class labels) --")
    by_folder = defaultdict(list)
    for r in rows:
        by_folder[r["folder"]].append(r)
    print(f"{'folder':50} {'videos':>6} {'broken':>6} {'minutes':>8} {'sources':>7}")
    for folder, rs in sorted(by_folder.items()):
        mins = sum(float(r["duration_s"] or 0) for r in rs if r["ok"]) / 60
        broken = sum(1 for r in rs if not r["ok"])
        sources = len({r["source_id"] for r in rs})
        print(f"{folder[:50]:50} {len(rs):>6} {broken:>6} {mins:>8.1f} {sources:>7}")

    def top(counter, n=6):
        return ", ".join(f"{k} x{v}" for k, v in counter.most_common(n))

    print("\n-- formats (readable files) --")
    print("resolution:", top(Counter(f"{r['width']}x{r['height']}" for r in ok)))
    print("fps:       ", top(Counter(str(r["fps"]) for r in ok)))
    print("codec:     ", top(Counter(r["codec"] for r in ok)))
    durs = sorted(float(r["duration_s"]) for r in ok if r["duration_s"])
    if durs:
        mid = durs[len(durs) // 2]
        print(f"clip length s: min {durs[0]:.1f}  median {mid:.1f}  max {durs[-1]:.1f}")

    if not args.no_hash:
        dupes = defaultdict(list)
        for r in rows:
            dupes[r["sha1"]].append(r["path"])
        groups = [g for g in dupes.values() if len(g) > 1]
        print(f"\n-- exact duplicates: {len(groups)} group(s) --")
        for g in groups[:10]:
            print("  " + "  ==  ".join(g))

    broken_rows = [r for r in rows if not r["ok"]]
    if broken_rows:
        print("\n-- broken files (first 10) --")
        for r in broken_rows[:10]:
            print(f"  {r['path']}: {r['error']}")

    if others:
        print("\n-- non-video files (label sheets, readmes...) --")
        for p in others[:20]:
            print("  " + str(p.relative_to(root)))
        if len(others) > 20:
            print(f"  ... and {len(others) - 20} more")

    print("\nsample source_id guesses (check these look right):")
    for r in rows[:5]:
        print(f"  {r['file']}  ->  {r['source_id']}")
    print(f"\nFull table written to {out}")


if __name__ == "__main__":
    main()