import os
import json
import pytest

HISTORY_PATH = os.path.join('models', 'history.log')
ALLOWED_REL_DROP = float(os.getenv('ALLOWED_ACCURACY_DROP', '0.15'))  # 15% default


def _load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    entries = []
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def test_relative_accuracy_regression_guard():
    """Ensure the latest run accuracy has not degraded more than allowed relative drop vs previous run.
    Skip if fewer than 2 recorded runs."""
    history = _load_history()
    if len(history) < 2:
        pytest.skip('Not enough runs to compute relative drop')
    prev, latest = history[-2], history[-1]
    prev_acc = prev.get('test_accuracy')
    latest_acc = latest.get('test_accuracy')
    assert prev_acc is not None and latest_acc is not None, 'Missing accuracy in history entries'
    if prev_acc == 0:
        pytest.skip('Previous accuracy is zero; relative drop undefined')
    rel_drop = (prev_acc - latest_acc) / prev_acc
    assert rel_drop <= ALLOWED_REL_DROP, (
        f"Relative accuracy drop {rel_drop:.3f} exceeds allowed {ALLOWED_REL_DROP:.3f} (prev={prev_acc:.4f}, latest={latest_acc:.4f})"
    )