#!/usr/bin/env python3
"""
Week 2 · Day 8 (Team A) — conformal partitions for the mentor-locked benchmark trio.

Re-partitions the Week-1 **unified v1.0** splits (leakage-safe, seed 42) into the Week-2
**Train / Calibration / Test / Zero-Day** conformal layout for **CIC-IoT2023, BoT-IoT, UNSW-NB15**
(mentor decision: TON_IoT + Edge-IIoTset set aside — their unified zero-day collapses to 1–2 rows).

Design (nothing is re-shuffled — we reuse the verified Week-1 assignment so exchangeability is preserved):
  * **Train = Week-1 train + val** (val folded in; both are known-classes → still known-only).
  * **Calibration / Test / Zero-Day = Week-1 splits unchanged.** Calibration is **KNOWN classes only**;
    the zero-day family is a full held-out attack class (Mirai / Theft / Worms+Shellcode).

Outputs (CSV v1 = 17 unified features + `label_multiclass`/`label_binary`/`label_family`):
  week2/partitions/<name>/{train,calibration,test,zeroday}.csv
  week2/partitions/<name>/partition_meta.json      (per-split class membership, known classes, zero-day)
  week2/reports/_generated/partition_summary.json  (cross-dataset summary for w2_01)

Run (venv, after the Week-1 unified pipeline): python week2/scripts/make_partitions.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "data" / "FROZEN"
GEN = BASE / "data" / "FROZEN" / "_generated"

SEED = 42
TRIO = ["CICIoT2023", "BoT-IoT", "UNSW-NB15"]
LABELS = ["label_multiclass", "label_binary", "label_family"]
# Week-2 conformal layout: train (=train+val), calibration, test, zeroday
SRC_FOR = {"train": ["train", "val"], "calibration": ["calibration"], "test": ["test"], "zeroday": ["zeroday"]}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(name: str) -> dict:
    src = DATA / name
    parts = {s: pd.read_parquet(src / f"{s}.parquet") for s in ["train", "val", "calibration", "test", "zeroday"]}
    outdir = OUT / name
    outdir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split, sources in SRC_FOR.items():
        df = pd.concat([parts[s] for s in sources], ignore_index=True)
        df.to_csv(outdir / f"{split}.csv", index=False)
        splits[split] = df

    feats = [c for c in splits["train"].columns if c not in LABELS]
    known = sorted(set(splits["train"]["label_multiclass"].unique()))
    zd_fams = sorted(set(splits["zeroday"]["label_multiclass"].unique())) if len(splits["zeroday"]) else []
    # invariants
    assert set(zd_fams).isdisjoint(known), f"{name}: zero-day family leaked into train"
    for s in ("calibration", "test"):
        assert set(splits[s]["label_multiclass"].unique()).isdisjoint(zd_fams), f"{name}: zero-day in {s}"

    meta = {
        "dataset": name, "seed": SEED, "n_features": len(feats), "features": feats,
        "layout": "train(=train+val) / calibration / test / zeroday",
        "known_classes": known, "n_known_classes": len(known), "zero_day_families": zd_fams,
        "rows": {s: int(len(v)) for s, v in splits.items()},
        "class_membership": {s: v["label_multiclass"].value_counts().sort_index().to_dict()
                             for s, v in splits.items()},
        "calibration_known_only": bool(set(splits["calibration"]["label_multiclass"].unique()).issubset(known)),
    }
    (outdir / "partition_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  {name}: train={len(splits['train']):,} cal={len(splits['calibration']):,} "
        f"test={len(splits['test']):,} zeroday={len(splits['zeroday']):,} | "
        f"{len(known)} known + zero-day {zd_fams}")
    return meta


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    log("Week-2 Day-8 partitions (CIC / BoT / UNSW) — Train/Calibration/Test/Zero-Day CSVs")
    summary = {name: run(name) for name in TRIO}
    (GEN / "partition_summary.json").write_text(
        json.dumps({n: {k: m[k] for k in ("rows", "n_known_classes", "zero_day_families",
                                          "calibration_known_only")} for n, m in summary.items()}, indent=1),
        encoding="utf-8")
    log(f"done -> week2/partitions/<name>/*.csv + partition_meta.json ({len(TRIO)} datasets)")


if __name__ == "__main__":
    main()
