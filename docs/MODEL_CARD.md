# Model Card: Legal Case Outcome Classifier

Last updated: 2025-09-30

## Overview

- Task: Multi-class classification of judgment outcomes from Indian court case text.
- Architecture: TF-IDF (1–3-grams) + AdaBoost(RandomForest). Logistic Regression baseline for reference.
- Calibration: Platt scaling attempted when feasible (sigmoid).

## Data (Ontology v1.1 / Curated Phase2)

- Rows (raw curated phase2): 1368
- Duplicate ratio: 0.137
- Curated Labels (post-demotion, pre-upsampling, 9): Other, Charge Sheet Quashed, Bail Granted, Labour / Wages / Termination, Contract Dispute, Acquittal, Divorce / Custody, Anticipatory Bail, Injunction Suit
- New Ontology Leaf: charge_sheet_quashed (mapped from curated label "Charge Sheet Quashed")
- Rare labels (<5) demoted to Other: Bribery / Prevention of Corruption Act, Domestic Violence Act
- Upsampling (optional) raises all minority classes to target (e.g., 50) for tail-focused training diagnostics; production inference still operates on original frequency distribution.

## Metrics

### Baseline (Pre-curation legacy coarse classes)
- Test Accuracy: 0.861
- Test Macro-F1: 0.284
- Weighted-F1: 0.815

### Curated Phase1 (12 classes, no upsampling)
- Accuracy: ~0.927
- Macro-F1 (boosted RF): ~0.281 (tails unstable)
- Macro-F1 (logreg baseline): ~0.452–0.625 depending on seed/test stratification

### Curated Phase2 (9 classes, no upsampling, demotion applied)
- Accuracy: ~0.920
- Macro-F1 (boosted RF pre-cal): ~0.287
- Logistic Regression macro-F1: ~0.625 (superior tail handling pre-balancing)

### Curated Phase2 + Upsampling (target minority=50)
- Accuracy (boosted RF): 0.988 (diagnostic; inflated due to balanced train/test partition)
- Macro-F1: 0.976 (diagnostic)
- Charge Sheet Quashed F1: 0.842 (pre-cal), 0.800 (post-cal)
- Logistic Regression Accuracy: 0.976 | Macro-F1: 0.959 | Charge Sheet Quashed F1: 0.750
- Selected Model: Boosted RF (higher macro-F1 under balanced diagnostic scenario)

### Head / Tail Diagnostic
- Head vs Tail macro-F1 converged after aggressive upsampling; unbiased holdout planned (see Roadmap).

NOTE: Upsampled metrics are for comparative diagnostic purposes only and should not be quoted as production performance. For production, rely on pre-upsampling macro-F1 and per-class F1 for supported labels.
- Drift: Jensen–Shannon divergence (/api/drift/compare) warn ≥0.10 alert ≥0.20
- Per-class F1 available (see metrics.json)
- Regression guard: relative accuracy drop ≤15% (tests enforce)

## Intended Use

Assistive legal outcome insight & precedent retrieval; not a substitute for professional legal advice.

## Limitations & Risks

- Class imbalance: production distribution still dominated by "Other"; diagnostic upsampling can mask real-world precision/recall.
- Tail volatility: minority label performance not yet validated on a held-out unbiased set post-upsampling.
- Calibration suppression: calibration skipped when min class support < threshold; if forced, can degrade tail F1.
- Ontology coverage: only one new leaf (charge_sheet_quashed) added; additional procedural outcomes pending evidence (support ≥ 25).
- Potential overfitting in aggressively balanced diagnostic runs (need neutral holdout split).

## Ethical Considerations

- Human-in-the-loop recommended; do not automate decisions.
- Monitor per-class performance for bias.
- Handle sensitive inputs carefully; enable log scrubbing where required.

## Versioning & Governance

- Run ID: 20250928201627_8909e349
- Previous run: 20250927195703_ada08c39
- Trained at (UTC): 2025-09-28T20:16:33.209875+05:30
- Artifacts: models/runs/<run_id>/ + latest copies in models/
- /api/version exposes latest model metadata.
- /api/drift/baseline for baseline distribution; /api/drift/compare for divergence.

## Reproducibility

### Data Assembly
1. Harvest / prepare raw: `python scripts/train_full_pipeline.py --skip-harvest` (or include harvest stage without the flag).
2. Enrichment + pattern mining (produces disposition counts & relabel file).
3. Apply relabels: `python scripts/apply_outcome_relabels.py`.
4. Build curated dataset (phase1): `python scripts/build_training_from_enriched.py`.
5. Phase2 curation (demotion): `python scripts/curate_labels_phase2.py`.
6. (Optional) Replace curated v1 with v2: copy `all_cases_curated_v2.csv` to `all_cases_curated.csv`.

### Training (Curated Mode)
Env (examples):
```
SET USE_CURATED_LABELS=1
SET CURATED_APPLY_PHASE2=1
SET CURATED_MIN_SUPPORT=5
SET UPSAMPLE_MAX_TARGET=50
SET UPSAMPLE_RATIO=0.4
SET UPSAMPLE_MIN_SUPPORT=5
SET SELECT_BEST_MODEL=1
python scripts/train_full_pipeline.py --skip-harvest
```

### Calibration Policy
- Skips when any train class support < CALIBRATION_MIN_SAMPLES (default 10).

### Model Selection
- If `SELECT_BEST_MODEL=1`, compares boosted RF vs Logistic Regression macro-F1; selects better with optional margin (`MODEL_SELECT_MARGIN`).

### Regenerate Model Card
- After training, run: `python scripts/generate_model_card.py` (future enhancement to include curated stats automatically).

## Changelog

- 2025-09-30: Ontology v1.1 (added charge_sheet_quashed). Curated labeling pipeline + phase2 demotion + upsampling diagnostics + head/tail metrics.
- 2025-09-28: Initial model card with legacy coarse classes.
