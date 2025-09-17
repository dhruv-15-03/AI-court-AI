# AI Court

Predict outcomes of Indian court cases using a boosted RandomForest text model with a simple Flask API.

## Setup

1) Python environment
- Python 3.10+ recommended
- Install dependencies:
  - On Windows PowerShell:
    - Create venv: `python -m venv .venv`
    - Activate: `.\\.venv\\Scripts\\Activate.ps1`
    - Install: `pip install -r requirements.txt`

2) Environment variables
- Copy `.env.example` to `.env` and fill values as needed:
  - `HUGGINGFACE_API_TOKEN` (optional) for summarization in the scraper
  - `MODEL_PATH` (optional) to override default `models/legal_case_classifier.pkl`

## Data

- Raw CSVs can be placed in project root or under `data/`.
- To scrape additional cases from Indian Kanoon:
  - `python -m src.ai_court.data.ingestion --query "cases on rape and murder" --pages 2`
  - This writes CSVs to `data/raw/`
- To normalize datasets into a unified schema:
  - `python -m src.ai_court.data.prepare_dataset`
  - This creates `data/processed/all_cases.csv`

## Train

- Quick start:
  - `python scripts/train_model.py`
  - If `data/processed/all_cases.csv` exists, trainer uses it; otherwise it auto-discovers CSVs in the repo.
- Model artifact saved to `models/legal_case_classifier.pkl`

## Serve API

- Start server: `python run_server.py`
- Endpoints:
  - `GET /api/questions` → questionnaire
  - `POST /api/analyze` → infer outcome; pass JSON with `case_type` and fields

## Project structure

- `src/ai_court/api/server.py` — Flask API
- `src/ai_court/model/legal_case_classifier.py` — training pipeline (TF-IDF + AdaBoost(RandomForest))
- `src/ai_court/scraper/legacy_kanoon.py` — Kanoon scraping (legacy)
- `src/ai_court/scraper/kanoon.py` — scraper wrapper module
- `src/ai_court/data/ingestion.py` — scraper wrapper → `data/raw/`
- `src/ai_court/data/prepare_dataset.py` — schema coercion → `data/processed/all_cases.csv`
- `models/` — saved model artifacts
- `logs/` — scraper/debug logs
- `scripts/` — utilities (train_model.py, smoke_client.py, cleanup_repo.py)

## Notes

- Do not commit secrets. Use `.env` and environment variables.
- With tiny classes, stratified splitting is disabled to avoid errors.
- Add more judgments to improve accuracy; keep the schema.
