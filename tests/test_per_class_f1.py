"""Per-class F1 regression guards.

Three independent tests:
1. test_per_class_f1_absolute_floor
   - Every class in the current metrics.json must be >= its configured floor.
   - Minorit-class (Acquittal) floor is intentionally separate from majority floors
     so that a bad retraining can never hide behind a high weighted-F1.

2. test_per_class_f1_regression_vs_previous_run
   - For classes present in both the latest and second-latest history.log entries,
     the relative F1 drop must not exceed ALLOWED_PER_CLASS_F1_DROP (default 25%).

3. test_metrics_metadata_consistency
   - models/metrics.json and models/metadata.json must agree on num_classes,
     classes list, run_id, test_macro_f1, and per_class_f1.
   - This catches the P0 issue of two files being written from different runs.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")
METADATA_PATH = os.path.join(ROOT, "models", "metadata.json")
HISTORY_PATH = os.path.join(ROOT, "models", "history.log")

# ---------------------------------------------------------------------------
# Configurable thresholds via env vars
# ---------------------------------------------------------------------------
# Absolute F1 floor for the Acquittal/most-sensitive minority class.
ACQUITTAL_FLOOR = float(os.getenv("ACQUITTAL_F1_FLOOR", "0.45"))
# Absolute F1 floor for all other classes.
DEFAULT_CLASS_FLOOR = float(os.getenv("DEFAULT_CLASS_F1_FLOOR", "0.60"))
# Maximum allowed relative F1 drop between consecutive runs, per class.
ALLOWED_PER_CLASS_DROP = float(os.getenv("ALLOWED_PER_CLASS_F1_DROP", "0.25"))

# Classes known to be safety-critical — each gets its own floor.
CRITICAL_CLASS_FLOORS: Dict[str, float] = {
    "Acquittal/Conviction Overturned": ACQUITTAL_FLOOR,
    "acquitted": ACQUITTAL_FLOOR,
    "Acquittal": ACQUITTAL_FLOOR,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics_json() -> Optional[dict]:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_metadata_json() -> Optional[dict]:
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    entries = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _per_class_f1_from_metrics(data: dict) -> Optional[Dict[str, float]]:
    """Extract per_class_f1 from a metrics.json dict."""
    return data.get("final_model", {}).get("per_class_f1")


def _per_class_f1_from_history_entry(entry: dict) -> Optional[Dict[str, float]]:
    """Extract per_class_f1 from a history.log entry."""
    return entry.get("per_class_f1")


# ---------------------------------------------------------------------------
# Test 1 — Absolute floor per class
# ---------------------------------------------------------------------------

def test_per_class_f1_absolute_floor():
    """Each class F1 in the current model must meet its configured minimum floor."""
    data = _load_metrics_json()
    if data is None:
        pytest.skip("models/metrics.json not found; skipping (pre-training environment)")

    per_class = _per_class_f1_from_metrics(data)
    if not per_class:
        pytest.skip("per_class_f1 not present in metrics.json; re-train to populate")

    failures = []
    for cls, f1 in per_class.items():
        floor = CRITICAL_CLASS_FLOORS.get(cls, DEFAULT_CLASS_FLOOR)
        if f1 < floor:
            failures.append(f"  {cls!r}: F1={f1:.4f} < floor={floor:.4f}")

    assert not failures, (
        "The following classes are below their minimum F1 floor:\n"
        + "\n".join(failures)
        + "\n\nFix: enable oversampling (MIN_CLASS_SAMPLES env var) or collect more data."
    )


# ---------------------------------------------------------------------------
# Test 2 — Relative regression vs previous run (history.log)
# ---------------------------------------------------------------------------

def test_per_class_f1_regression_vs_previous_run():
    """No class may drop more than ALLOWED_PER_CLASS_DROP relative F1 between runs."""
    history = _load_history()

    if len(history) < 2:
        pytest.skip("Fewer than 2 history entries; cannot compare runs")

    prev_entry = history[-2]
    latest_entry = history[-1]

    prev_pcf1 = _per_class_f1_from_history_entry(prev_entry)
    latest_pcf1 = _per_class_f1_from_history_entry(latest_entry)

    if not prev_pcf1 or not latest_pcf1:
        pytest.skip("per_class_f1 not recorded in both history entries; re-train to populate")

    failures = []
    for cls in prev_pcf1:
        if cls not in latest_pcf1:
            # Class disappeared entirely — that is a regression
            failures.append(
                f"  {cls!r}: present in previous run but missing from latest run"
            )
            continue
        prev_f1 = prev_pcf1[cls]
        latest_f1 = latest_pcf1[cls]
        if prev_f1 == 0:
            continue  # Relative drop undefined; skip
        rel_drop = (prev_f1 - latest_f1) / prev_f1
        if rel_drop > ALLOWED_PER_CLASS_DROP:
            failures.append(
                f"  {cls!r}: {prev_f1:.4f} → {latest_f1:.4f} "
                f"(drop={rel_drop:.2%} > allowed={ALLOWED_PER_CLASS_DROP:.0%})"
            )

    assert not failures, (
        "Per-class F1 regression detected between the two most recent runs:\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Test 3 — metrics.json ↔ metadata.json consistency
# ---------------------------------------------------------------------------

def test_metrics_metadata_consistency():
    """metrics.json and metadata.json must describe the same training run."""
    metrics_data = _load_metrics_json()
    meta_data = _load_metadata_json()

    if metrics_data is None or meta_data is None:
        pytest.skip("One or both artifact files missing; skipping consistency check")

    fm = metrics_data.get("final_model", {})

    mismatches = []

    # run_id must agree
    metrics_run = fm.get("run_id")
    meta_run = meta_data.get("run_id")
    if metrics_run and meta_run and metrics_run != meta_run:
        mismatches.append(
            f"  run_id: metrics.json={metrics_run!r} vs metadata.json={meta_run!r}"
        )

    # num_classes must agree
    metrics_nc = fm.get("num_classes")
    meta_nc = meta_data.get("num_classes")
    if metrics_nc is not None and meta_nc is not None and metrics_nc != meta_nc:
        mismatches.append(
            f"  num_classes: metrics.json={metrics_nc} vs metadata.json={meta_nc}"
        )

    # classes list must agree (order-insensitive)
    metrics_cls = sorted(fm.get("classes") or [])
    meta_cls = sorted(meta_data.get("classes") or [])
    if metrics_cls and meta_cls and metrics_cls != meta_cls:
        mismatches.append(
            f"  classes: metrics.json={metrics_cls} vs metadata.json={meta_cls}"
        )

    # test_macro_f1 must agree (within floating point tolerance)
    metrics_mf1 = fm.get("test_macro_f1")
    meta_mf1 = meta_data.get("test_macro_f1")
    if metrics_mf1 is not None and meta_mf1 is not None:
        if abs(metrics_mf1 - meta_mf1) > 1e-6:
            mismatches.append(
                f"  test_macro_f1: metrics.json={metrics_mf1} vs metadata.json={meta_mf1}"
            )

    # per_class_f1 must agree
    metrics_pcf1 = fm.get("per_class_f1") or {}
    meta_pcf1 = meta_data.get("per_class_f1") or {}
    if metrics_pcf1 and meta_pcf1:
        for cls in set(metrics_pcf1) | set(meta_pcf1):
            mv = metrics_pcf1.get(cls)
            dv = meta_pcf1.get(cls)
            if mv is None or dv is None or abs(mv - dv) > 1e-6:
                mismatches.append(
                    f"  per_class_f1[{cls!r}]: metrics.json={mv} vs metadata.json={dv}"
                )

    assert not mismatches, (
        "models/metrics.json and models/metadata.json are out of sync "
        "(likely written from different training runs):\n"
        + "\n".join(mismatches)
        + "\n\nFix: re-run training so both files are written atomically from the same run."
    )
