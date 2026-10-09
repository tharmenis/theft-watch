#!/usr/bin/env python3
"""
Theft Watch - Phase 4, steps 9-11: train the small classifier, pick the alert
threshold, and report how well it works.

The classifier is a logistic regression: it learns one weight per feature number
and adds them up into a 0-1 "theft score". Small, fast, hard to over-fit with
~1,000 examples, and easy to move to the edge box later.

Reads   data/labels/windows.csv
        data/features/<backbone>/{features.npy, features_flip.npy, index.csv, meta.json}
Writes  models/behaviour/<run>/   classifier + settings + report + list of mistakes
        (and an MLflow run, if mlflow is installed)

Normal use - train, tune on VALIDATION, report validation only:
    docker compose exec trainer python training/train_classifier.py
The one-time honest score on the TEST split (step 11):
    docker compose exec trainer python training/train_classifier.py --final
Every --final run is logged in models/behaviour/test_log.csv, so you can see
how often the test set has been looked at.

Options:
    --target-recall 0.8   threshold is set so validation catches >= 80% of theft windows
    --model mlp           a small neural network instead of logistic regression
    --name my-run         name for this run's folder
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("scikit-learn is missing. In the trainer container run:  pip install scikit-learn mlflow")


# ----------------------------------------------------------------- data

def load(windows_path: Path, feat_dir: Path):
    with open(windows_path, newline="") as f:
        windows = list(csv.DictReader(f))
    with open(feat_dir / "index.csv", newline="") as f:
        index = list(csv.DictReader(f))
    feats = np.load(feat_dir / "features.npy").astype(np.float32)
    flip_path = feat_dir / "features_flip.npy"
    flips = np.load(flip_path).astype(np.float32) if flip_path.exists() else None
    meta = json.loads((feat_dir / "meta.json").read_text())

    by_id = {w["window_id"]: w for w in windows}
    missing = [r["window_id"] for r in index if r["window_id"] not in by_id]
    if missing or len(index) != len(windows):
        sys.exit(f"Features ({len(index)} windows) do not match windows.csv ({len(windows)}). "
                 f"Re-run extract_features.py after prepare_windows.py.")
    rows = [by_id[r["window_id"]] for r in index]
    has_flip = np.array([r["has_flip"] == "1" for r in index])
    if "random" in str(meta.get("weights", "")):
        print("WARNING: these features came from RANDOM weights (test mode) - results mean nothing.")
    return rows, feats, flips, has_flip, meta


def split_idx(rows, name):
    return np.array([i for i, r in enumerate(rows) if r["split"] == name], dtype=int)


# ----------------------------------------------------------------- model

def fit(model_kind, X, y, X_val, y_val, seed):
    """Try a few strengths of regularisation, keep the one best on validation."""
    scaler = StandardScaler().fit(X)
    Xs, Xv = scaler.transform(X), scaler.transform(X_val)
    tried = []
    if model_kind == "logreg":
        grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        make = lambda c: LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    else:
        grid = [1e-3, 1e-2, 1e-1, 1.0]
        make = lambda a: MLPClassifier(hidden_layer_sizes=(256,), alpha=a, max_iter=400,
                                       early_stopping=True, random_state=seed)
    best = None
    for p in grid:
        clf = make(p)
        if model_kind == "mlp":           # MLP has no class_weight: oversample positives
            pos = np.where(y == 1)[0]
            reps = max(1, int(round((y == 0).sum() / max(1, len(pos)))) - 1)
            Xs_fit = np.concatenate([Xs] + [Xs[pos]] * reps)
            y_fit = np.concatenate([y] + [y[pos]] * reps)
        else:
            Xs_fit, y_fit = Xs, y
        clf.fit(Xs_fit, y_fit)
        ap = average_precision_score(y_val, clf.predict_proba(Xv)[:, 1])
        tried.append((p, ap))
        if best is None or ap > best[1]:
            best = (p, ap, clf)
    return scaler, best[2], best[0], tried


def pick_threshold(scores, y, target_recall):
    """Highest threshold that still catches >= target_recall of the theft windows."""
    pos = np.sort(scores[y == 1])[::-1]
    if len(pos) == 0:
        sys.exit("No concealment windows in validation - cannot set a threshold.")
    k = int(np.ceil(target_recall * len(pos))) - 1
    return float(pos[min(k, len(pos) - 1)])


# ----------------------------------------------------------------- evaluation

def groups_of_overlapping(items):
    """items: list of (start, end, payload) sorted by start -> list of groups."""
    groups = []
    for it in items:
        if groups and it[0] < groups[-1][-1][1]:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def union_seconds(ranges):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(ranges):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def evaluate(rows, idx, scores, thr):
    y = np.array([int(rows[i]["target"]) for i in idx])
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    res = {
        "windows": len(idx), "theft_windows": int(y.sum()), "threshold": round(thr, 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 3) if 0 < y.sum() < len(y) else None,
        "roc_auc": round(float(roc_auc_score(y, scores)), 3) if 0 < y.sum() < len(y) else None,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "window_recall": round(tp / max(1, tp + fn), 3),
        "window_precision": round(tp / max(1, tp + fp), 3),
    }

    per_video_pos, per_video_neg = defaultdict(list), defaultdict(list)
    for j, i in enumerate(idx):
        r = rows[i]
        item = (float(r["start_s"]), float(r["end_s"]), (scores[j], r))
        (per_video_pos if r["target"] == "1" else per_video_neg)[r["video"]].append(item)

    # events: overlapping theft windows in one video = one event; caught if any window fires
    events = caught = 0
    for v, items in per_video_pos.items():
        for g in groups_of_overlapping(sorted(items, key=lambda t: t[0])):
            events += 1
            caught += any(sc >= thr for _, _, (sc, _) in g)
    res["events"] = events
    res["events_caught"] = caught
    res["event_recall"] = round(caught / max(1, events), 3)

    # false alerts: overlapping firing normal windows in one video = one alert
    by_scene = defaultdict(lambda: {"alerts": 0, "seconds": 0.0, "windows": 0, "fired": 0})
    for v, items in per_video_neg.items():
        items.sort(key=lambda t: t[0])
        scene = items[0][2][1]["scene"]
        d = by_scene[scene]
        d["seconds"] += union_seconds([(s, e) for s, e, _ in items])
        d["windows"] += len(items)
        fired = [it for it in items if it[2][0] >= thr]
        d["fired"] += len(fired)
        d["alerts"] += len(groups_of_overlapping(fired))
    total_alerts = sum(d["alerts"] for d in by_scene.values())
    total_hours = sum(d["seconds"] for d in by_scene.values()) / 3600
    res["false_alerts"] = total_alerts
    res["normal_minutes"] = round(total_hours * 60, 1)
    res["false_alerts_per_hour"] = round(total_alerts / total_hours, 1) if total_hours else None
    res["by_scene"] = {
        sc: {"normal_windows": d["windows"], "windows_fired": d["fired"],
             "share_fired": round(d["fired"] / max(1, d["windows"]), 3),
             "false_alerts": d["alerts"], "minutes": round(d["seconds"] / 60, 1)}
        for sc, d in sorted(by_scene.items())}
    return res


def mistakes(rows, idx, scores, thr, n=15):
    out = []
    for j, i in enumerate(idx):
        r, s = rows[i], float(scores[j])
        if r["target"] == "1" and s < thr:
            kind = "missed theft"
        elif r["target"] == "0" and s >= thr:
            kind = "false alert"
        else:
            continue
        out.append({"kind": kind, "score": round(s, 3), "video": r["video"],
                    "start_s": r["start_s"], "end_s": r["end_s"], "scene": r["scene"],
                    "window_id": r["window_id"]})
    # most confident mistakes first
    out.sort(key=lambda m: -abs(m["score"] - thr))
    return out


def print_report(name, res):
    c = res["confusion"]
    print(f"\n--- {name.upper()} ({res['windows']} windows, {res['theft_windows']} theft) ---")
    print(f"threshold (score needed to alert): {res['threshold']}")
    print(f"PR-AUC {res['pr_auc']}   ROC-AUC {res['roc_auc']}   "
          f"(1.0 = perfect ranking; PR-AUC of a coin toss here = "
          f"{res['theft_windows'] / max(1, res['windows']):.2f})")
    print("confusion (windows):        said THEFT   said normal")
    print(f"   really theft             {c['tp']:>10}   {c['fn']:>11}")
    print(f"   really normal            {c['fp']:>10}   {c['tn']:>11}")
    print(f"theft windows caught: {res['window_recall']:.0%}    "
          f"alerts that were real theft: {res['window_precision']:.0%}")
    print(f"theft EVENTS caught: {res['events_caught']}/{res['events']} ({res['event_recall']:.0%})")
    print(f"false alerts: {res['false_alerts']} in {res['normal_minutes']} min of normal footage "
          f"= ~{res['false_alerts_per_hour']} per hour (very rough at this sample size)")
    print("room check - share of NORMAL windows that fired, per scene:")
    for sc, d in res["by_scene"].items():
        print(f"   {sc:20} {d['windows_fired']:>4}/{d['normal_windows']:<4} "
              f"({d['share_fired']:.0%})   false alerts {d['false_alerts']}")
    shares = {sc: d["share_fired"] for sc, d in res["by_scene"].items() if d["normal_windows"] >= 5}
    if len(shares) >= 2:
        hi, lo = max(shares, key=shares.get), min(shares, key=shares.get)
        if shares[hi] >= 0.10 and shares[hi] >= 3 * max(shares[lo], 0.01):
            print(f"   >> WARNING: normal behaviour in '{hi}' fires far more often than in '{lo}'."
                  f" The model may be recognising the ROOM, not the action.")
        else:
            print("   >> no strong sign of the room shortcut")


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default="data/labels/windows.csv")
    ap.add_argument("--features", default="data/features/mvit_v2_s")
    ap.add_argument("--out", default="models/behaviour")
    ap.add_argument("--model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--target-recall", type=float, default=0.80)
    ap.add_argument("--no-flip", action="store_true", help="ignore mirrored training copies")
    ap.add_argument("--final", action="store_true", help="also score the TEST split (step 11)")
    ap.add_argument("--name", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_dir = Path(args.features)
    rows, feats, flips, has_flip, fmeta = load(Path(args.windows), feat_dir)
    tr, va, te = (split_idx(rows, s) for s in ("train", "val", "test"))
    y_all = np.array([int(r["target"]) for r in rows])
    uses = Counter(r.get("allowed_use", "") for r in rows)
    if "" in uses:
        print("NOTE: some windows have no allowed_use (register not found when preparing windows).")

    X_tr, y_tr = feats[tr], y_all[tr]
    if flips is not None and not args.no_flip:
        fl = tr[has_flip[tr]]
        X_tr = np.concatenate([X_tr, flips[fl]])
        y_tr = np.concatenate([y_tr, y_all[fl]])
    print(f"training on {len(y_tr)} examples ({int(y_tr.sum())} theft) "
          f"incl. mirrored copies; validating on {len(va)} windows")

    t0 = time.time()
    scaler, clf, param, tried = fit(args.model, X_tr, y_tr, feats[va], y_all[va], args.seed)
    print(f"regularisation tried (setting -> validation PR-AUC): "
          + ", ".join(f"{p:g}->{a:.3f}" for p, a in tried) + f"   kept {param:g}")

    score = lambda idx: clf.predict_proba(scaler.transform(feats[idx]))[:, 1]
    s_tr, s_va = score(tr), score(va)
    thr = pick_threshold(s_va, y_all[va], args.target_recall)

    report = {"train": evaluate(rows, tr, s_tr, thr), "val": evaluate(rows, va, s_va, thr)}
    print_report("train (the model has seen these - expect it to look good)", report["train"])
    print_report("validation", report["val"])
    if args.final:
        s_te = score(te)
        report["test"] = evaluate(rows, te, s_te, thr)
        print_report("TEST - the honest score", report["test"])

    # ---- save ----
    run = args.name or time.strftime("%Y%m%d-%H%M%S") + f"-{args.model}"
    out = Path(args.out) / run
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "classifier": clf}, out / "classifier.joblib")
    if args.model == "logreg":   # plain numbers too, for the edge export later
        np.savez(out / "head.npz", mean=scaler.mean_, scale=scaler.scale_,
                 coef=clf.coef_[0], intercept=clf.intercept_)
    metadata = {
        "name": f"theft-watch-behaviour-{run}", "task": "concealment vs normal",
        "classes": ["normal", "concealment"], "threshold": round(thr, 4),
        "threshold_rule": f"validation window recall >= {args.target_recall}",
        "model": args.model, "regularisation": param,
        "backbone": fmeta, "windows_file": args.windows,
        "allowed_use_of_training_data": dict(uses),
        "status": "INTERNAL TESTING ONLY - trained on public data; must not ship",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    errs = [dict(m, split="val") for m in mistakes(rows, va, s_va, thr)]
    if args.final:
        errs += [dict(m, split="test") for m in mistakes(rows, te, s_te, thr)]
    if errs:
        with open(out / "mistakes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(errs[-1].keys()))
            w.writeheader()
            w.writerows(errs)

    if args.final:
        log = Path(args.out) / "test_log.csv"
        first = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["time", "run", "event_recall", "false_alerts_per_hour", "pr_auc"])
            t = report["test"]
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), run, t["event_recall"],
                        t["false_alerts_per_hour"], t["pr_auc"]])
        with open(log) as f:
            n = sum(1 for _ in f) - 1
        if n > 1:
            print(f"\nNOTE: the test set has now been scored {n} times. Each extra look makes "
                  f"its score a little less honest - tune on validation, not test.")

    try:
        import mlflow
        mlflow.set_experiment("theft-watch-behaviour")
        with mlflow.start_run(run_name=run):
            mlflow.log_params({"model": args.model, "regularisation": param,
                               "target_recall": args.target_recall, "threshold": round(thr, 4),
                               "backbone": fmeta.get("backbone"), "flip": not args.no_flip,
                               "train_examples": len(y_tr), "final": args.final})
            for split, r in report.items():
                for k in ("pr_auc", "roc_auc", "window_recall", "window_precision",
                          "event_recall", "false_alerts_per_hour"):
                    if r.get(k) is not None:
                        mlflow.log_metric(f"{split}_{k}", r[k])
            mlflow.log_artifacts(str(out))
        print("\nlogged to MLflow (view with:  mlflow ui --host 0.0.0.0)")
    except ImportError:
        print("\n(mlflow not installed - results are only in the run folder)")

    print(f"\nsaved to {out}/  ({time.time() - t0:.0f} s)")
    if errs:
        m = errs[0]
        print(f"watch the mistakes listed in {out}/mistakes.csv, e.g.:")
        print(f"   mpv --start={m['start_s']} --end={m['end_s']} data/raw/{m['video']}")


if __name__ == "__main__":
    main()