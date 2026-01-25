"""Generate or refresh docs/MODEL_CARD.md from latest training artifacts.

Usage:
  python scripts/generate_model_card.py [--fail-on-missing]

Reads models/metadata.json and models/metrics.json then renders a concise
model card with sections: Overview, Data, Metrics, Intended Use, Limitations,
Ethics, Versioning/Governance, Reproducibility, Changelog.

Idempotent: re-writes file in place. Safe for CI usage after training.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_CARD_PATH = os.path.join(DOCS_DIR, "MODEL_CARD.md")


def load_json(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def render(metadata: dict | None, metrics: dict | None) -> str:
    now = datetime.now(timezone.utc).date().isoformat()
    fm = (metrics or {}).get('final_model', {}) if metrics else {}
    test_accuracy = fm.get('test_accuracy')
    macro_f1 = fm.get('test_macro_f1')
    weighted_f1 = fm.get('test_weighted_f1') or fm.get('train_f1_weighted')
    cv_mean = metrics and metrics.get('final_model', {}).get('cv_macro_f1_mean')
    cv_std = metrics and metrics.get('final_model', {}).get('cv_macro_f1_std')
    duplicate_ratio = metadata and metadata.get('duplicate_ratio')
    train_f1 = metadata and metadata.get('train_f1_weighted')
    total_rows = metadata and metadata.get('dataset_rows')
    classes = metadata and metadata.get('classes') or []
    class_dist = metadata and metadata.get('class_distribution') or {}
    run_id = metadata and metadata.get('run_id')
    previous_run = metadata and metadata.get('previous_run')
    trained_at = metadata and metadata.get('trained_at')

    distribution_lines = []
    if class_dist:
        total = sum(class_dist.values()) or 1
        for label, count in sorted(class_dist.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total
            distribution_lines.append(f"  - {label}: {count} ({pct:.1f}%)")
    distribution_block = ("\n" + "\n".join(distribution_lines)) if distribution_lines else " (not available)"

    parts = [
        "# Model Card: Legal Case Outcome Classifier\n",
        f"Last updated: {now}\n",
        "## Overview\n",
        "- Task: Multi-class classification of judgment outcomes from Indian court case text.",
        "- Architecture: TF-IDF (1–3-grams) + AdaBoost(RandomForest). Logistic Regression baseline for reference.",
        "- Calibration: Platt scaling attempted when feasible (sigmoid).",
        "\n## Data\n",
        f"- Rows: {total_rows if total_rows is not None else 'unknown'}",
        f"- Duplicate ratio: {duplicate_ratio:.3f}" if duplicate_ratio is not None else "- Duplicate ratio: unknown",
        f"- Classes ({len(classes)}): {', '.join(classes)}" if classes else "- Classes: unknown",
        f"- Distribution:{distribution_block}\n",
        "## Metrics (latest run)\n",
        f"- Test Accuracy: {test_accuracy:.3f}" if isinstance(test_accuracy, (int, float)) else "- Test Accuracy: unknown",
    f"- Train F1 (weighted): {train_f1:.3f}" if isinstance(train_f1, (int, float)) else "- Train F1 (weighted): unknown",
    f"- Test Macro-F1: {macro_f1:.3f}" if isinstance(macro_f1, (int, float)) else "- Test Macro-F1: unknown",
    f"- Test Weighted-F1: {weighted_f1:.3f}" if isinstance(weighted_f1, (int, float)) else "- Test Weighted-F1: unknown",
        (f"- CV macro-F1 (train split): {cv_mean:.3f} ± {cv_std:.3f}" if isinstance(cv_mean, (int, float)) and isinstance(cv_std, (int, float)) else "- CV macro-F1: unavailable"),
    "- Drift: Jensen–Shannon divergence (/api/drift/compare) warn ≥0.10 alert ≥0.20",
    ("- Per-class F1 available (see metrics.json)" if fm.get('per_class_f1') else "- Per-class F1: will appear after next retrain"),
        "- Regression guard: relative accuracy drop ≤15% (tests enforce)\n",
        "## Intended Use\n",
        "Assistive legal outcome insight & precedent retrieval; not a substitute for professional legal advice.\n",
        "## Limitations & Risks\n",
        "- Class imbalance (Other dominates).",
        "- Possible label noise from heuristic normalization.",
        "- Domain / temporal drift (new laws, precedents).",
        "- Small minority classes may have unstable metrics.\n",
        "## Ethical Considerations\n",
        "- Human-in-the-loop recommended; do not automate decisions.",
        "- Monitor per-class performance for bias.",
        "- Handle sensitive inputs carefully; enable log scrubbing where required.\n",
        "## Versioning & Governance\n",
        f"- Run ID: {run_id if run_id else 'unknown'}",
        f"- Previous run: {previous_run if previous_run else 'none'}",
        f"- Trained at (UTC): {trained_at if trained_at else 'unknown'}",
        "- Artifacts: models/runs/<run_id>/ + latest copies in models/",
        "- /api/version exposes latest model metadata.",
        "- /api/drift/baseline for baseline distribution; /api/drift/compare for divergence.\n",
        "## Reproducibility\n",
        "1. Prepare dataset: python -m src.ai_court.data.prepare_dataset",
        "2. Train: python scripts/train_model.py",
        "3. (Optional) Build search index: python scripts/build_search_index.py",
        "4. Regenerate model card: python scripts/generate_model_card.py\n",
        "## Changelog\n",
        "- Automated model card generation script added.",
    ]
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fail-on-missing', action='store_true', help='Exit non-zero if artifacts missing')
    args = parser.parse_args()

    metadata = load_json(os.path.join(MODELS_DIR, 'metadata.json'))
    metrics = load_json(os.path.join(MODELS_DIR, 'metrics.json'))
    if args.fail_on_missing and (metadata is None or metrics is None):
        raise SystemExit("Required artifacts missing (metadata.json / metrics.json)")

    os.makedirs(DOCS_DIR, exist_ok=True)
    content = render(metadata, metrics)
    with open(MODEL_CARD_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Model card written to {MODEL_CARD_PATH}")


if __name__ == '__main__':
    main()
