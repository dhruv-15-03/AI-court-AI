# Model Card: Legal Case Outcome Classifier

Last updated: 2026-01-04

## Overview

- Task: Multi-class classification of judgment outcomes from Indian court case text.
- Architecture: TF-IDF (1–3-grams) + AdaBoost(RandomForest). Logistic Regression baseline for reference.
- Calibration: Platt scaling attempted when feasible (sigmoid).

## Data

- Rows: 10838
- Duplicate ratio: 0.000
- Classes (3): Acquittal/Conviction Overturned, Relief Denied/Dismissed, Relief Granted/Convicted
- Distribution:
  - Relief Granted/Convicted: 7663 (70.7%)
  - Relief Denied/Dismissed: 1901 (17.5%)
  - Acquittal/Conviction Overturned: 1274 (11.8%)

## Metrics (latest run)

- Test Accuracy: 0.918
- Train F1 (weighted): 1.000
- Test Macro-F1: 0.827
- Test Weighted-F1: 0.910
- CV macro-F1 (train split): 0.807 ± 0.012
- Drift: Jensen–Shannon divergence (/api/drift/compare) warn ≥0.10 alert ≥0.20
- Per-class F1 available (see metrics.json)
- Regression guard: relative accuracy drop ≤15% (tests enforce)

## Intended Use

Assistive legal outcome insight & precedent retrieval; not a substitute for professional legal advice.

## Limitations & Risks

- Class imbalance (Other dominates).
- Possible label noise from heuristic normalization.
- Domain / temporal drift (new laws, precedents).
- Small minority classes may have unstable metrics.

## Ethical Considerations

- Human-in-the-loop recommended; do not automate decisions.
- Monitor per-class performance for bias.
- Handle sensitive inputs carefully; enable log scrubbing where required.

## Versioning & Governance

- Run ID: 20251012162202_b8f3da89
- Previous run: 20251012161402_d76bc17e
- Trained at (UTC): 2025-10-12T16:22:07.263961+05:30
- Artifacts: models/runs/<run_id>/ + latest copies in models/
- /api/version exposes latest model metadata.
- /api/drift/baseline for baseline distribution; /api/drift/compare for divergence.

## Reproducibility

1. Prepare dataset: python -m src.ai_court.data.prepare_dataset
2. Train: python scripts/train_model.py
3. (Optional) Build search index: python scripts/build_search_index.py
4. Regenerate model card: python scripts/generate_model_card.py

## Changelog

- Automated model card generation script added.
