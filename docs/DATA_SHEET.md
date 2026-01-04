# Data Sheet: AI Court Dataset

Last updated: 2025-11-05

## Motivation
Support outcome prediction and precedent retrieval for Indian court cases, enabling faster exploration of similar cases.

## Composition
- Records: consolidated from harvested Indian Kanoon pages and curated CSVs.
- Fields (normalized): `id`, `title`, `url`, `case_type`, `case_summary`, `judgment` (mapped to coarse classes).
- Size: Latest training dataset has 10,838 rows (see `models/metadata.json`). Search index counts live in `models/search_index.pkl` metadata.

## Collection Process
- Harvester `scripts/kanoon_harvest.py` collects pages by queries into CSVs (`data/raw`), with retries and logging to `logs/legal_scraper.log`.
- Optional enrichment via Hugging Face Inference API (`scripts/enrich_summaries_with_hf.py`).
- Dataset builder (`src/ai_court/data/prepare_dataset.py`) normalizes to a unified schema and filters low-quality/unknown entries.

## Preprocessing & Labeling
- Text normalization suitable for TF‑IDF features.
- Outcome normalization via heuristic mapping to three coarse classes: Acquittal/Conviction Overturned; Relief Denied/Dismissed; Relief Granted/Convicted.

## Uses
- Training/prediction of outcomes and case similarity retrieval.
- Not for making legal decisions; human-in-the-loop recommended.

## Distribution
- Local CSVs under `data/` and artifacts in `models/`.
- No redistribution of source documents included; scraped content remains subject to the source site’s terms.
- Class distribution (latest training set):
	- Relief Granted/Convicted: 7,663
	- Relief Denied/Dismissed: 1,901
	- Acquittal/Conviction Overturned: 1,274

See also distribution plots in `docs/judgement_distribution.png` and `docs/case_type_distribution.png`.

## Maintenance
- Version info via `/version`.
- Rebuild search index after adding or enriching data.
- Retrain periodically; monitor `/api/drift/compare` (warn ≥0.10, alert ≥0.20).

## Risks and Limitations
- Class imbalance and label noise.
- Potential scraping errors or site changes.
- Summary generation quality varies by HF model and timeouts.

## Ethical Considerations
- Respect source terms; avoid storing PII unnecessarily.
- Provide transparency via model card and per-class metrics.
