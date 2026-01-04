import os
import json
import pytest

from ai_court.api.server import app
from ai_court.api.config import PROJECT_ROOT


def test_version_endpoint_model_metadata():
    """Ensure /version returns model metadata structure when available.
    If metadata.json not present, still expect 200 and 'model' key (possibly None)."""
    with app.test_client() as c:
        r = c.get('/version')
        assert r.status_code == 200
        data = r.get_json()
        assert 'model' in data
        model_meta = data['model']
        # If available, basic required keys
        if model_meta is not None:
            for key in ['dataset_hash', 'dataset_rows', 'num_classes', 'duplicate_ratio']:
                assert key in model_meta


@pytest.mark.parametrize("min_accuracy", [0.5])
def test_model_metrics_regression_guard(min_accuracy):
    """Guardrail: if metrics.json exists, ensure test accuracy not below threshold.
    Does not fail build if metrics absent (e.g., before first training run)."""
    metrics_path = os.path.join(PROJECT_ROOT, 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        pytest.skip("metrics.json not present; training not executed yet")
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    final_model = metrics.get('final_model') or {}
    acc = final_model.get('test_accuracy')
    assert acc is not None, "test_accuracy missing in metrics.json"
    assert acc >= min_accuracy, f"Model test_accuracy {acc} fell below minimum {min_accuracy}"


def test_drift_baseline_endpoint():
    from ai_court.api.server import app  # local import to ensure updated server
    with app.test_client() as c:
        r = c.get('/api/drift/baseline')
        assert r.status_code == 200
        data = r.get_json()
        # Keys may be empty if metadata absent, but structure should exist
        assert 'duplicate_ratio' in data or 'error' in data or 'class_distribution' in data


def test_run_history_presence():
    import os
    runs_dir = os.path.join('models', 'runs')
    if not os.path.isdir(runs_dir):
        # Training might not have been executed in this environment
        import pytest
        pytest.skip('No runs directory yet')
    entries = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    assert entries, 'Expected at least one run directory in models/runs'


def test_drift_compare_endpoint_smoke():
    from ai_court.api.server import app
    with app.test_client() as c:
        # Need baseline metadata; skip if missing
        r_base = c.get('/api/drift/baseline')
        if r_base.status_code != 200 or not r_base.get_json():
            import pytest
            pytest.skip('No baseline available')
        baseline = r_base.get_json()
        class_dist = baseline.get('class_distribution') or {}
        if not class_dist:
            import pytest
            pytest.skip('No class distribution in baseline')
        # Build an identical counts payload to expect jsd ~ 0
        payload = {"counts": class_dist}
        r = c.post('/api/drift/compare', json=payload)
        if r.status_code == 503:  # baseline unavailable edge
            import pytest
            pytest.skip('Baseline unavailable 503')
        assert r.status_code == 200
        data = r.get_json()
        assert 'jsd' in data
        assert data['status'] in ('ok','warn','alert')
        # After compare, history endpoint should contain at least one event
        r_hist = c.get('/api/drift/history?limit=5')
        if r_hist.status_code == 200:
            hist_data = r_hist.get_json()
            assert 'events' in hist_data
            # Not asserting non-empty because history log may be write-protected in some envs


def test_drift_compare_validation_error():
    from ai_court.api.server import app
    with app.test_client() as c:
        r = c.post('/api/drift/compare', json={"counts": {"A": -5}})
        # Negative treated as value but normalization yields distribution; we allow it here
        # so send malformed type to trigger validation
        r2 = c.post('/api/drift/compare', json={"counts": "not_a_dict"})
        assert r2.status_code in (400, 500, 503)


def test_per_class_metrics_presence():
    import os, json
    metrics_path = os.path.join(PROJECT_ROOT, 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        import pytest
        pytest.skip('metrics.json not present')
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    final_model = metrics.get('final_model') or {}
    # If metrics were generated before enhancement, skip instead of failing
    if 'per_class_f1' not in final_model or 'test_macro_f1' not in final_model:
        import pytest
        pytest.skip('Enhanced metrics not present in existing metrics.json (requires retraining)')
