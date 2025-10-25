# Data Sheet: AI Court Dataset

Last updated: 2025-09-23

## Motivation
Support outcome prediction and precedent retrieval for Indian court cases, enabling faster exploration of similar cases.

## Composition
- Records: consolidated from harvested Indian Kanoon pages and curated CSVs.
- Fields (normalized): `id`, `title`, `url`, `case_type`, `case_data`/`case_summary`, `judgment` (mapped to coarse classes).
- Size: Varies by harvest; see `data/processed/all_cases.csv` and `models/search_index.pkl` metadata for counts.

## Collection Process
- Scraper (`src/ai_court/data/ingestion.py`) with retries and multiple selectors.
- Optional enrichment via Hugging Face Inference API for summaries (resumable with checkpoints).
- Dataset builder (`src/ai_court/data/prepare_dataset.py`) normalizes to unified schema.

## Preprocessing & Labeling
- Text normalization and lemmatization with legal term preservation.
- Outcome normalization via heuristic mapping in `normalize_outcome()`.

## Uses
- Training/prediction of outcomes and case similarity retrieval.
- Not for making legal decisions; human-in-the-loop recommended.

## Distribution
- Local CSVs under `data/` and artifacts in `models/`.
- No redistribution of source documents included; scraped content remains subject to the source site’s terms.

## Maintenance
- Version info via `/api/version`.
- Rebuild index after adding or enriching data.
- Retrain periodically to address drift.

## Risks and Limitations
- Class imbalance and label noise.
- Potential scraping errors or site changes.
- Summary generation quality varies by HF model and timeouts.

## Ethical Considerations
- Respect source terms; avoid storing PII unnecessarily.
- Provide transparency via model card and per-class metrics.
