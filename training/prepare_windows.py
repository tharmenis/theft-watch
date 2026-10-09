#!/usr/bin/env python3
"""
Theft Watch - Phase 3, steps 5-6: cut videos into labelled windows and split them.

Reads
    data/interim/inventory.csv     (from inspect_dataset.py)
    data/labels/events.csv         (your marked events)
    docs/dataset_register.csv      (optional: adds dataset_id / allowed_use)
Writes
    data/labels/splits.csv         one row per source video -> train / val / test
                                   (created once, then REUSED so the test set never changes)
    data/labels/windows.csv        one row per window: video, start, end, label, split ...
    data/interim/samples/...       (only with --samples N) short MP4s to eyeball the labels

No video files are cut here: a "window" is just a time range of a source video.
The feature step reads those frames straight from the original file.

How a window gets its label (defaults: 4 s windows every 1 s):
    concealment / hard_negative : the window covers >= 75% of that event
    excluded (not used)         : touches an event but covers less than that,
                                  or touches an 'unclear' event,
                                  or is within 0.5 s of an event (label timing slack)
    normal                      : everything else

Run inside the trainer container, from the repo root:
    docker compose exec trainer python training/prepare_windows.py
    docker compose exec trainer python training/prepare_windows.py --samples 5
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

POSITIVE = {"concealment"}            # what the model should alert on
SPECIAL = {"concealment", "hard_negative", "unclear"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def num(x: str):
    x = (x or "").strip().replace(",", ".")
    return float(x) if x else None


def load_register(path: Path) -> list:
    """Return [(folder_prefix_relative_to_data_raw, dataset_id, allowed_use)], longest first."""
    if not path.exists():
        return []
    out = []
    for r in read_csv(path):
        folder = (r.get("folder") or "").strip().rstrip("/")
        folder = re.sub(r"^(\./)?data/raw/?", "", folder)
        out.append((folder, r.get("dataset_id", "").strip(), r.get("allowed_use", "").strip()))
    return sorted(out, key=lambda t: -len(t[0]))


def match_register(video: str, reg: list):
    for folder, ds, use in reg:
        if folder == "" or video == folder or video.startswith(folder + "/"):
            return ds, use
    return "", ""


def make_windows(dur: float, length: float, stride: float) -> list:
    if dur < length:
        return [(0.0, round(dur, 3))] if dur >= 0.75 * length else []
    starts, s = [], 0.0
    while s + length <= dur + 1e-6:
        starts.append(round(s, 3))
        s += stride
    if dur - (starts[-1] + length) > 0.5 * stride:      # cover the tail too
        starts.append(round(dur - length, 3))
    return [(s, round(s + length, 3)) for s in starts]


def label_window(ws, we, events, cover, margin):
    """events: list of (start, end, action) with action in SPECIAL. Returns (label, reason)."""
    hit = None
    for es, ee, a in events:
        ov = max(0.0, min(we, ee) - max(ws, es))
        near = ov > 0 or (ws - margin < ee and we + margin > es)
        if not near:
            continue
        if a == "unclear":
            return None, "touches unclear event"
        need = cover * min(ee - es, we - ws)
        if ov >= need - 1e-6:
            if hit and hit != a:
                return None, "covers two different events"
            hit = a
        else:
            return None, "partial overlap with event"
    return (hit or "normal"), ""


def why_missed(es, ee, a, evs, dur, args) -> str:
    """Best guess at why no window was labelled with this event."""
    others = [(s, t, b) for s, t, b in evs if (s, t, b) != (es, ee, a)]
    near = [o for o in others if o[0] < ee + args.window and o[1] > es - args.window]
    if ee > dur + 0.05:
        return "ends after the video ends"
    if near:
        o = min(near, key=lambda o: abs(o[0] - es))
        return (f"another event ({o[2]} {o[0]}-{o[1]}) is too close: every window that "
                f"covers this one also clips that one")
    if ee - es < 0.5:
        return "very short event at a spot no window lines up with - check the times"
    return "no window covers enough of it (near the video edge?)"


def assign_splits(videos: dict, old: dict, ratios, seed: int) -> dict:
    """videos: {group: stratum}. Keeps any existing assignment; splits only new groups,
    stratified by stratum (class folder) so every split gets both kinds of video."""
    out = {g: old[g] for g in videos if g in old}
    new = defaultdict(list)
    for g, stratum in videos.items():
        if g not in out:
            new[stratum].append(g)
    rng = random.Random(seed)
    for stratum, gs in sorted(new.items()):
        gs.sort(key=natural_key)
        rng.shuffle(gs)
        n = len(gs)
        n_test = max(1, round(ratios[2] * n)) if n >= 3 else 0
        n_val = max(1, round(ratios[1] * n)) if n >= 3 else 0
        for i, g in enumerate(gs):
            out[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def export_samples(rows: list, root: Path, out_dir: Path, n: int, seed: int) -> None:
    try:
        import av
    except ImportError:
        print("  (PyAV missing - skipping samples)")
        return
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for label, rs in sorted(by_label.items()):
        for r in rng.sample(rs, min(n, len(rs))):
            dst = out_dir / label / f"{r['window_id']}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            ws, we = float(r["start_s"]), float(r["end_s"])
            with av.open(str(root / r["video"])) as inp, av.open(str(dst), "w") as out:
                vin = inp.streams.video[0]
                rate = vin.average_rate or 30
                vout = out.add_stream("mpeg4", rate=rate)
                vout.width, vout.height = vin.codec_context.width, vin.codec_context.height
                vout.pix_fmt, vout.bit_rate = "yuv420p", 2_000_000
                for fr in inp.decode(video=0):
                    if fr.time is None or fr.time < ws:
                        continue
                    if fr.time >= we:
                        break
                    for p in vout.encode(fr.reformat(format="yuv420p")):
                        out.mux(p)
                for p in vout.encode():
                    out.mux(p)
    print(f"  samples written to {out_dir}/<label>/  (open them in mpv/VLC)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--inventory", default="data/interim/inventory.csv")
    ap.add_argument("--events", default="data/labels/events.csv")
    ap.add_argument("--register", default="docs/dataset_register.csv")
    ap.add_argument("--splits", default="data/labels/splits.csv")
    ap.add_argument("--out", default="data/labels/windows.csv")
    ap.add_argument("--window", type=float, default=4.0, help="window length, seconds")
    ap.add_argument("--stride", type=float, default=1.0, help="step between windows, seconds")
    ap.add_argument("--cover", type=float, default=0.75, help="share of an event a window must cover")
    ap.add_argument("--margin", type=float, default=0.5, help="normal windows keep this far from events")
    ap.add_argument("--merge-gap", type=float, default=1.0,
                    help="merge events of the same kind closer than this, seconds (0 = off)")
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test share of videos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=0, help="export N sample windows per label")
    args = ap.parse_args()

    inv = {r["path"]: r for r in read_csv(Path(args.inventory))}
    events = read_csv(Path(args.events))
    reg = load_register(Path(args.register))
    ratios = [float(x) for x in args.ratios.split(",")]

    # ---- group events per video ----
    per_video = defaultdict(list)
    whole = {}
    groups = {}
    for e in events:
        v = e["video"].strip()
        if v not in inv:
            sys.exit(f"{v} is in events.csv but not in the inventory. Run --check first.")
        a = e["action"].strip().lower()
        s, t = num(e["start_s"]), num(e["end_s"])
        groups[v] = (e.get("source") or "").strip() or v   # optional 'source' column
        if s is None and t is None:
            whole[v] = a
        elif a in SPECIAL:
            per_video[v].append((s, t, a))
    videos = sorted(set(whole) | set(per_video), key=natural_key)

    # ---- merge events of the same kind that are (almost) back to back ----
    merged_count = 0
    if args.merge_gap > 0:
        for v, evs in per_video.items():
            evs.sort()
            out = []
            for s, t, a in evs:
                if out and out[-1][2] == a and s - out[-1][1] <= args.merge_gap:
                    out[-1] = (out[-1][0], max(out[-1][1], t), a)
                    merged_count += 1
                else:
                    out.append((s, t, a))
            per_video[v] = out

    # ---- splits: per source group, frozen once written ----
    split_path = Path(args.splits)
    old = {r["group"]: r["split"] for r in read_csv(split_path)} if split_path.exists() else {}
    strata = {groups[v]: inv[v]["folder"] for v in videos}
    split_of = assign_splits(strata, old, ratios, args.seed)
    write_csv(split_path,
              [{"group": g, "split": split_of[g], "stratum": strata[g]}
               for g in sorted(split_of, key=natural_key)],
              ["group", "split", "stratum"])
    reused = sum(1 for g in strata if g in old)

    # ---- windows ----
    rows, excluded, missed_events, unregistered = [], Counter(), [], set()
    for v in videos:
        dur = float(inv[v]["duration_s"] or 0)
        ds, use = match_register(v, reg)
        if reg and not ds:
            unregistered.add(v.split("/")[0])
        evs = sorted(per_video.get(v, []))
        hits = Counter()
        for i, (ws, we) in enumerate(make_windows(dur, args.window, args.stride)):
            if v in whole:
                label = whole[v] if whole[v] != "unclear" else None
                reason = "whole video unclear"
            else:
                label, reason = label_window(ws, we, evs, args.cover, args.margin)
            if label is None:
                excluded[reason] += 1
                continue
            if label != "normal":
                for es, ee, a in evs:
                    if a == label and min(we, ee) > max(ws, es):
                        hits[(es, ee)] += 1
            rows.append({
                "window_id": f"w{len(rows):06d}",
                "video": v, "start_s": ws, "end_s": we,
                "label": label, "target": 1 if label in POSITIVE else 0,
                "split": split_of[groups[v]], "scene": Path(inv[v]["folder"]).name,
                "group": groups[v], "dataset_id": ds, "allowed_use": use,
            })
        for es, ee, a in evs:
            if a != "unclear" and hits[(es, ee)] == 0:
                missed_events.append({"video": v, "start_s": es, "end_s": ee, "action": a,
                                      "length_s": round(ee - es, 2),
                                      "video_length_s": round(dur, 2),
                                      "reason": why_missed(es, ee, a, evs, dur, args)})

    fields = ["window_id", "video", "start_s", "end_s", "label", "target",
              "split", "scene", "group", "dataset_id", "allowed_use"]
    write_csv(Path(args.out), rows, fields)

    # ---- summary ----
    print("\n=== THEFT WATCH WINDOWS ===")
    print(f"videos: {len(videos)}   windows kept: {len(rows)}   excluded: {sum(excluded.values())}")
    print(f"window {args.window}s every {args.stride}s, event cover >= {args.cover:.0%}, "
          f"margin {args.margin}s")
    vids_per_split = Counter(split_of[groups[v]] for v in videos)
    print("\n-- per split --")
    print(f"{'split':6} {'videos':>6} {'windows':>8} {'concealment':>12} {'normal':>7} {'hard_neg':>9}")
    for sp in ("train", "val", "test"):
        c = Counter(r["label"] for r in rows if r["split"] == sp)
        print(f"{sp:6} {vids_per_split[sp]:>6} {sum(c.values()):>8} {c['concealment']:>12} "
              f"{c['normal']:>7} {c['hard_negative']:>9}")
    print("\n-- normal windows by scene (for the 'did it learn the room?' check) --")
    c = Counter((r["split"], r["scene"]) for r in rows if r["label"] == "normal")
    for (sp, sc), n in sorted(c.items()):
        print(f"  {sp:6} {sc:20} {n}")
    if excluded:
        print("\n-- excluded windows --")
        for reason, n in excluded.most_common():
            print(f"  {reason}: {n}")
    if args.merge_gap > 0:
        print(f"\nmerged {merged_count} event(s) into a neighbour (gap <= {args.merge_gap}s)")
    if missed_events:
        miss_path = Path("data/interim/missed_events.csv")
        write_csv(miss_path, missed_events, list(missed_events[0].keys()))
        print(f"\n-- events with NO window covering them ({len(missed_events)}), "
              f"full list in {miss_path} --")
        for reason, n in Counter(m["reason"].split(" (")[0] for m in missed_events).most_common():
            print(f"  {n:3}  {reason}")
        for m in missed_events[:5]:
            print(f"  e.g. {m['video']} {m['start_s']}-{m['end_s']}: {m['reason']}")
    print(f"\nsplits: {split_path} ({reused} groups reused, {len(strata) - reused} newly assigned)")
    if not reg:
        print(f"WARNING: {args.register} not found - dataset_id/allowed_use left empty. "
              f"Put the register there, or pass --register <path>")
    elif unregistered:
        print(f"WARNING: no register row for: {sorted(unregistered)}")
    uses = Counter(r["allowed_use"] for r in rows)
    print(f"allowed_use of kept windows: {dict(uses)}")
    print(f"windows written to {args.out}")

    if args.samples:
        export_samples(rows, Path(args.raw), Path("data/interim/samples"), args.samples, args.seed)


if __name__ == "__main__":
    main()