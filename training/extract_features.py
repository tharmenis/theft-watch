#!/usr/bin/env python3
"""
Theft Watch - Phase 4, step 8: turn every window into a feature vector.

A pretrained video model (default: torchvision MViT-V2-S, trained on the
Kinetics-400 action dataset) looks at 16 frames from each 4 s window and
outputs 768 numbers that summarise the motion in it. We drop its last layer
(the one that names 400 everyday actions) and keep those numbers. The model
itself is frozen: nothing is trained here.

Reads   data/labels/windows.csv  and the videos under data/raw
Writes  data/features/<backbone>/features.npy        (windows x 768, float16)
        data/features/<backbone>/features_flip.npy   (same, mirrored video; train only)
        data/features/<backbone>/index.csv           (row i <-> window_id)
        data/features/<backbone>/meta.json           (settings, versions, timings)

Run inside the trainer container, from the repo root:
    docker compose exec trainer python training/extract_features.py
First run downloads the pretrained weights (~130 MB) into the torch cache.

Options worth knowing:
    --backbone r2plus1d_18   smaller/older model (512 numbers, 112 px) if memory is tight
    --batch 2                lower this if you see "CUDA out of memory"
    --limit 20               only do the first 20 windows (quick test)
"""
import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import av
    import torch
    import torchvision
    from torchvision.models import video as tvv
except ImportError as e:
    sys.exit(f"Missing package ({e}). Run this inside the trainer container.")

BACKBONES = {
    # name: (builder, weights enum, input size, mean, std, feature size, head attribute)
    "mvit_v2_s": (tvv.mvit_v2_s, tvv.MViT_V2_S_Weights.KINETICS400_V1, 224,
                  (0.45, 0.45, 0.45), (0.225, 0.225, 0.225), 768, "head"),
    "r2plus1d_18": (tvv.r2plus1d_18, tvv.R2Plus1D_18_Weights.KINETICS400_V1, 112,
                    (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989), 512, "fc"),
}


def build_model(name: str, pretrained: bool, device):
    builder, weights, size, mean, std, dim, head = BACKBONES[name]
    model = builder(weights=weights if pretrained else None)
    setattr(model, head, torch.nn.Identity())        # keep the features, drop the 400 classes
    model.eval().to(device)
    return model, size, mean, std, dim


def decode_video(path: Path, size: int):
    """Decode the whole (short) video once, resized to size x size. Returns (times, frames)."""
    times, frames = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            if fr.time is None:
                continue
            times.append(fr.time)
            # Squash the full frame to a square instead of cropping its sides away,
            # so a hand at the edge of the picture is not cut off.
            frames.append(fr.to_ndarray(width=size, height=size, format="rgb24"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.asarray(times), np.stack(frames)          # (N,), (N, H, W, 3) uint8


def pick_frames(times, frames, start, end, n):
    """n frames spread evenly across [start, end)."""
    step = (end - start) / n
    targets = start + step * (np.arange(n) + 0.5)
    idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
    return frames[idx]                                   # (n, H, W, 3)


def to_tensor(clips, mean, std, device):
    """list of (T,H,W,3) uint8 -> (B,3,T,H,W) normalised float."""
    x = torch.from_numpy(np.stack(clips)).to(device)    # B,T,H,W,3
    x = x.permute(0, 4, 1, 2, 3).float().div_(255)      # B,3,T,H,W
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1, 1)
    return (x - m) / s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default="data/labels/windows.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--backbone", default="mvit_v2_s", choices=sorted(BACKBONES))
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per window")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-flip", action="store_true", help="skip mirrored copies of train windows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--random-weights", action="store_true", help="testing only: no download")
    args = ap.parse_args()

    with open(args.windows, newline="") as f:
        windows = list(csv.DictReader(f))
    if args.limit:
        windows = windows[:args.limit]
    if not windows:
        sys.exit("windows.csv is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU visible - this will be very slow. Check training/check_gpu.py.")
    use_fp16 = device.type == "cuda"
    model, size, mean, std, dim = build_model(args.backbone, not args.random_weights, device)
    print(f"backbone {args.backbone} on {torch.cuda.get_device_name(0) if use_fp16 else 'CPU'}, "
          f"{len(windows)} windows, {args.frames} frames @ {size}px, batch {args.batch}, "
          f"{'fp16' if use_fp16 else 'fp32'}")

    out_dir = Path(args.out) / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = np.zeros((len(windows), dim), dtype=np.float16)
    flips = np.zeros((len(windows), dim), dtype=np.float16)
    do_flip = [not args.no_flip and w["split"] == "train" for w in windows]

    by_video = defaultdict(list)
    for i, w in enumerate(windows):
        by_video[w["video"]].append(i)

    t0, done = time.time(), 0
    if use_fp16:
        torch.cuda.reset_peak_memory_stats()
    pending = []                                          # (row, clip, is_flip)

    def flush():
        nonlocal pending
        if not pending:
            return
        x = to_tensor([c for _, c, _ in pending], mean, std, device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=use_fp16,
                                                    dtype=torch.float16):
            y = model(x).float().cpu().numpy()
        for (row, _, is_flip), vec in zip(pending, y):
            (flips if is_flip else feats)[row] = vec
        pending = []

    for vi, (video, rows) in enumerate(by_video.items(), 1):
        times, frames = decode_video(Path(args.raw) / video, size)
        for row in rows:
            w = windows[row]
            clip = pick_frames(times, frames, float(w["start_s"]), float(w["end_s"]), args.frames)
            pending.append((row, clip, False))
            if do_flip[row]:
                pending.append((row, np.ascontiguousarray(clip[:, :, ::-1]), True))
            if len(pending) >= args.batch:
                flush()
        flush()
        done += len(rows)
        if vi % 10 == 0 or vi == len(by_video):
            rate = (time.time() - t0) / done
            left = rate * (len(windows) - done)
            print(f"  {vi}/{len(by_video)} videos, {done}/{len(windows)} windows, "
                  f"{rate:.2f} s/window, ~{left / 60:.1f} min left", flush=True)

    elapsed = time.time() - t0
    bad = int(np.isnan(feats.astype(np.float32)).any(axis=1).sum())
    np.save(out_dir / "features.npy", feats)
    if any(do_flip):
        np.save(out_dir / "features_flip.npy", flips)
    with open(out_dir / "index.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "window_id", "has_flip"])
        for i, w in enumerate(windows):
            wr.writerow([i, w["window_id"], int(do_flip[i])])
    meta = {
        "backbone": args.backbone,
        "weights": "random (TEST ONLY)" if args.random_weights
                   else str(BACKBONES[args.backbone][1]),
        "frames": args.frames, "input_size": size, "resize": "full frame squashed to square",
        "feature_dim": dim, "windows": len(windows), "flipped_windows": int(sum(do_flip)),
        "fp16": use_fp16, "seconds": round(elapsed, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if use_fp16 else None,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "python": platform.python_version(), "windows_file": args.windows,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== THEFT WATCH FEATURES ===")
    print(f"windows: {len(windows)}  (+{sum(do_flip)} mirrored train copies)")
    print(f"feature size: {dim}   time: {elapsed / 60:.1f} min   "
          f"peak GPU memory: {meta['peak_gpu_mb']} MB")
    f32 = feats.astype(np.float32)
    print(f"value check: mean {f32.mean():.3f}  std {f32.std():.3f}  rows with NaN: {bad}")
    print(f"written to {out_dir}/")
    if bad:
        print("WARNING: some rows are NaN - tell Claude; usually an fp16 overflow.")


if __name__ == "__main__":
    main()