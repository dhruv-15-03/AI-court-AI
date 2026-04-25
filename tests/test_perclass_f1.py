"""Per-class F1 regression guard.

Reads ``models/metrics.json`` (written by Trainer.run_training_pipeline)
and asserts that no individual class's F1 score has regressed below its
defined floor.

WHY THIS EXISTS
---------------
The top-level macro-F1 guard can mask a collapsed minority class:
e.g. if ``Acquittal/Conviction Overturned`` drops from 0.64 → 0.20
but ``Relief Granted`` stays at 0.96, the macro average only moves
from 0.83 → 0.70, which may still pass a 20% relative-drop guard.
This test catches it directly.

THRESHOLDS
----------
Set conservatively below the current known-good values so that normal
run-to-run variance does not fail the build, but a real regression does.

  Class                              Known-good    Floor
  ─────────────────────────────────  ──────────    ─────
  Acquittal/Conviction Overturned    0.6357        0.50
  Relief Denied/Dismissed            0.8833        0.72
  Relief Granted/Convicted           0.9621        0.88

Override via environment variables (e.g. in CI):
  FLOOR_F1_ACQUITTAL=0.50
  FLOOR_F1_RELIEF_DENIED=0.72
  FLOOR_F1_RELIEF_GRANTED=0.88
"""

from __future__ import annotations

import json
import os

import pytest

METRICS_PATH = os.path.join("models", "metrics.json")

# Floor thresholds — overridable via environment variables
FLOORS: dict[str, float] = {
    "Acquittal/Conviction Overturned": float(os.getenv("FLOOR_F1_ACQUITTAL", "0.50")),
    "Relief Denied/Dismissed":        float(os.getenv("FLOOR_F1_RELIEF_DENIED", "0.72")),
    "Relief Granted/Convicted":       float(os.getenv("FLOOR_F1_RELIEF_GRANTED", "0.88")),
}


def _load_per_class_f1() -> dict[str, float] | None:
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("final_model", {}).get("per_class_f1")
    except Exception:
        return None


def test_per_class_f1_not_missing():
    """metrics.json must exist and contain per_class_f1 after any training run."""
    if not os.path.exists(METRICS_PATH):
        pytest.skip("metrics.json not present; training not executed yet")
    per_class = _load_per_class_f1()
    assert per_class is not None, (
        "per_class_f1 key missing from metrics.json['final_model']. "
        "Re-run training with the updated Trainer to populate it."
    )
    assert len(per_class) > 0, "per_class_f1 is empty"


@pytest.mark.parametrize("label,floor", list(FLOORS.items()))
def test_per_class_f1_floor(label: str, floor: float):
    """Each critical class must maintain F1 above its floor threshold."""
    if not os.path.exists(METRICS_PATH):
        pytest.skip("metrics.json not present; training not executed yet")

    per_class = _load_per_class_f1()
    if per_class is None:
        pytest.skip("per_class_f1 not present in metrics.json (pre-P0 artifact)")

    if label not in per_class:
        # Label schema may differ between runs; skip rather than fail
        pytest.skip(f"Class '{label}' not in current model classes: {list(per_class.keys())}")

    actual = per_class[label]
    assert actual >= floor, (
        f"Class '{label}' F1 regressed: {actual:.4f} < floor {floor:.4f}. "
        "Check class imbalance, label normalisation, or recent data changes."
    )


def test_acquittal_not_dominated_by_relief_granted():
    """Specific guard: Acquittal F1 must be within 40pp of Relief Granted F1.

    A gap larger than this means the minority class is effectively ignored
    by the model despite class weighting, which is a signal to re-enable
    SMOTE (ENABLE_SMOTE=1) or collect more acquittal training samples.
    """
    if not os.path.exists(METRICS_PATH):
        pytest.skip("metrics.json not present")

    per_class = _load_per_class_f1()
    if per_class is None:
        pytest.skip("per_class_f1 not present in metrics.json")

    acquittal_f1 = per_class.get("Acquittal/Conviction Overturned")
    rg_f1 = per_class.get("Relief Granted/Convicted")

    if acquittal_f1 is None or rg_f1 is None:
        pytest.skip("Required classes not found in per_class_f1")

    gap = rg_f1 - acquittal_f1
    assert gap <= 0.40, (
        f"Acquittal F1 ({acquittal_f1:.4f}) is {gap:.4f} below Relief Granted F1 ({rg_f1:.4f}). "
        "Minority class is severely under-represented. Set ENABLE_SMOTE=1 or MIN_CLASS_SAMPLES=300."
    )
