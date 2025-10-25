import os
import json
import pytest

HISTORY_PATH = os.path.join('models', 'history.log')
ALLOWED_REL_DROP = float(os.getenv('ALLOWED_MACRO_F1_DROP', '0.20'))  # allow 20% relative macro-F1 drop


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


def test_relative_macro_f1_regression_guard():
    """Skip if fewer than 2 runs or macro-F1 not yet recorded."""
    history = _load_history()
    if len(history) < 2:
        pytest.skip('Not enough runs for macro-F1 regression check')
    prev, latest = history[-2], history[-1]
    prev_macro = prev.get('test_macro_f1')
    latest_macro = latest.get('test_macro_f1')
    if prev_macro is None or latest_macro is None:
        pytest.skip('Macro-F1 not present in history entries')
    if prev_macro == 0:
        pytest.skip('Previous macro-F1 zero; relative drop undefined')
    rel_drop = (prev_macro - latest_macro) / prev_macro
    assert rel_drop <= ALLOWED_REL_DROP, (
        f"Relative macro-F1 drop {rel_drop:.3f} exceeds allowed {ALLOWED_REL_DROP:.3f} (prev={prev_macro:.4f}, latest={latest_macro:.4f})"
    )
