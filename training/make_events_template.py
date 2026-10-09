#!/usr/bin/env python3
"""
Theft Watch - Phase 3, step 4: create and check the events file.

Default mode: builds data/labels/events.csv from data/interim/inventory.csv
  - videos in a folder named like 'normal'      -> one row, action=normal, whole video
  - videos in a folder named like 'shoplifting' -> one row, action=concealment,
                                                   start_s/end_s EMPTY for you to fill in
  - videos whose resolution differs from --keep-res are left out (default 640x480)
  Refuses to overwrite an existing events file (use --force).

Check mode (--check): reads your filled-in events file and reports problems.
  It never changes the file.

Run inside the trainer container, from the repo root:
    docker compose exec trainer python training/make_events_template.py
    docker compose exec trainer python training/make_events_template.py --check

Events file columns:
    video      path relative to data/raw (e.g. mypack/shoplifting/shoplifting-3.mp4)
    start_s    event start in seconds   (empty = whole video)
    end_s      event end in seconds     (empty = whole video)
    action     concealment | normal | unclear | hard_negative
    note       free text
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ACTIONS = {"concealment", "normal", "unclear", "hard_negative"}
FIELDS = ["video", "start_s", "end_s", "action", "note"]


def load_inventory(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"{path} not found. Run training/inspect_dataset.py first.")
    with open(path, newline="") as f:
        return {r["path"]: r for r in csv.DictReader(f)}


def class_of(folder: str) -> str:
    name = Path(folder).name.lower()
    if "normal" in name:
        return "normal"
    if "shoplift" in name or "theft" in name:
        return "concealment"
    return ""


def make(inv: dict, out: Path, keep_res: str, force: bool) -> None:
    if out.exists() and not force:
        sys.exit(f"{out} already exists - not overwriting your labels. "
                 f"Use --check to check it, or --force to start over.")
    rows, skipped_res, skipped_other = [], [], []
    for path, r in sorted(inv.items(), key=lambda kv: natural_key(kv[0])):
        if r["ok"] != "True":
            continue
        res = f"{r['width']}x{r['height']}"
        if keep_res and res != keep_res:
            skipped_res.append(path)
            continue
        cls = class_of(r["folder"])
        if not cls:
            skipped_other.append(path)
            continue
        rows.append({"video": path, "start_s": "", "end_s": "", "action": cls,
                     "note": "whole video" if cls == "normal" else ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["action"] for r in rows)
    print(f"Wrote {out}: {len(rows)} rows  ({dict(c)})")
    print(f"Left out {len(skipped_res)} video(s) not at {keep_res}.")
    if skipped_other:
        print(f"Left out {len(skipped_other)} video(s) in folders that are neither "
              f"'normal' nor 'shoplifting', e.g. {skipped_other[0]}")
    print("Next: fill in start_s and end_s for every 'concealment' row.")


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def check(inv: dict, path: Path) -> None:
    if not path.exists():
        sys.exit(f"{path} not found.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing_cols = [c for c in FIELDS[:4] if rows and c not in rows[0]]
    if missing_cols:
        sys.exit(f"Missing column(s): {missing_cols}. Expected: {FIELDS}")

    problems, todo = [], []
    per_video = {}
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        v, a = r["video"].strip(), r["action"].strip().lower()
        s, e = r["start_s"].strip().replace(",", "."), r["end_s"].strip().replace(",", ".")
        where = f"line {i} ({v})"
        if v not in inv:
            problems.append(f"{where}: video not in inventory (typo, or re-run the inventory?)")
            continue
        if a not in ACTIONS:
            problems.append(f"{where}: unknown action '{a}' (use {sorted(ACTIONS)})")
            continue
        dur = float(inv[v]["duration_s"] or 0)
        if not s and not e:
            if a == "concealment":
                todo.append(f"{where}: concealment without start/end time")
            per_video.setdefault(v, []).append((None, None, a))
            continue
        try:
            s_f, e_f = float(s), float(e)
        except ValueError:
            problems.append(f"{where}: start_s/end_s must be numbers (got '{s}', '{e}')")
            continue
        if not (0 <= s_f < e_f):
            problems.append(f"{where}: need 0 <= start < end (got {s_f}, {e_f})")
        elif e_f > dur + 0.5:
            problems.append(f"{where}: end {e_f}s is after the video ends ({dur:.1f}s)")
        elif e_f - s_f > 10:
            problems.append(f"{where}: event lasts {e_f - s_f:.1f}s - longer than expected, check it")
        per_video.setdefault(v, []).append((s_f, e_f, a))

    for v, evs in per_video.items():
        whole = [a for s, e, a in evs if s is None]
        if whole and len(evs) > 1:
            problems.append(f"{v}: has a whole-video row AND other rows - keep only one kind")
        timed = sorted((s, e) for s, e, a in evs if s is not None)
        for (s1, e1), (s2, e2) in zip(timed, timed[1:]):
            if s2 < e1:
                problems.append(f"{v}: events overlap ({s1}-{e1} and {s2}-{e2})")

    c = Counter(r["action"].strip().lower() for r in rows)
    print(f"{path}: {len(rows)} rows, {len(per_video)} videos  {dict(c)}")
    print(f"still to fill in: {len(todo)}   problems: {len(problems)}")
    for line in todo[:15]:
        print("  TODO   " + line)
    if len(todo) > 15:
        print(f"  ... and {len(todo) - 15} more")
    for line in problems[:30]:
        print("  CHECK  " + line)
    if not todo and not problems:
        print("All good - ready for the clip cutting step.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", default="data/interim/inventory.csv")
    ap.add_argument("--out", default="data/labels/events.csv")
    ap.add_argument("--keep-res", default="640x480",
                    help="only include videos at this resolution ('' = all)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    inv = load_inventory(Path(args.inventory))
    if args.check:
        check(inv, Path(args.out))
    else:
        make(inv, Path(args.out), args.keep_res, args.force)


if __name__ == "__main__":
    main()