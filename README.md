# 🏛️ AI Court

**AI-powered Indian legal case classifier and precedent search engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![Legal-BERT](https://img.shields.io/badge/Model-Legal--BERT-green.svg)](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
[![Cases](https://img.shields.io/badge/Cases-6.8K+-brightgreen.svg)](#current-metrics)
[![Accuracy](https://img.shields.io/badge/Accuracy-60.24%25-yellow.svg)](#model-performance)

> **Production-ready ML pipeline** for predicting Indian court case outcomes and retrieving similar legal precedents using Legal-BERT transformers and semantic search.

---

## 📊 Current Metrics

**Last Updated:** October 7, 2025 - 02:15 AM

| Metric | Value | Status |
|--------|-------|--------|
| **Total Cases** | 6,882 | 🟢 Growing at ~1K/hour |
| **Database Size** | 51.66 MB | ✅ Healthy |
| **Model Accuracy** | 60.24% | ⚠️ Training on 413 cases |
| **Next Milestone** | 10,000 | ⏳ 28 min ETA |
| **Collection Status** | Active | 🟢 Unlimited mode |

**Quick Check:** `python quick_status.py` | **Full Dashboard:** `python metrics_dashboard.py`

---

## Setup

1) Python environment
- Python 3.10+ recommended
## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env`:
```bash
MODEL_PATH=models/legal_case_classifier.pkl
SEARCH_INDEX_PATH=models/search_index.pkl
HUGGINGFACE_API_TOKEN=<optional_for_summaries>
```

### 3. Run the System

```powershell
# Quick status check
python quick_status.py

# Full metrics dashboard
python metrics_dashboard.py

# Start API server
python run_server.py

# Start data collection (unlimited mode)
python scripts/continuous_collector.py

# Train model
python scripts/pipeline/batch_trainer.py --batch_size 1000 --force
```

---

## 📚 Data Collection

### Current Database
- **Location:** `data/legal_cases_10M.db` (SQLite)
- **Size:** 51.66 MB (6,882 cases)
- **Collection Rate:** ~1,000 cases/hour
- **Target:** 100,000 cases (unlimited mode)

### Continuous Collector (PRODUCTION)

**Status:** 🟢 RUNNING - Unlimited mode with 161 search queries

```powershell
# Start collector
python scripts/continuous_collector.py

# Monitor progress
python quick_status.py

# Check logs
tail -f logs/legal_scraper.log
```

**Features:**
- 161 unique search queries (IPC sections, crimes, courts, outcomes, years)
- 12 parallel workers
- Random page starts (1-10) for variety
- Hash-based deduplication (SHA256)
- Optimized delays (0.08-0.2s)
- Deep search (up to 30 pages per query)

### Search Query Categories
- **35+ IPC Sections:** 302, 376, 420, 498A, 304, 307, etc.
- **30+ Crime Types:** Murder, rape, fraud, cyber crime, dowry death, etc.
- **15+ Courts:** Supreme Court, High Courts (Delhi, Bombay, Madras, etc.)
- **20+ Outcomes:** Bail, conviction, acquittal, dismissed, allowed, etc.
- **10+ Years:** 2015-2024 (recent cases prioritized)
- **20+ Legal Terms:** NDPS, POCSO, dying declaration, circumstantial evidence, etc.

---

## 🤖 Model Training

### Model #1 (Current)
- **Training Cases:** 413
- **Accuracy:** 60.24% ⚠️
- **F1 Weighted:** 45.29%
- **F1 Macro:** 18.80%
- **Device:** CUDA (RTX 4050 6GB)
- **Model:** Legal-BERT (nlpaueb/legal-bert-base-uncased)

### Train Model #2 (Recommended at 10K cases)

```powershell
# Train with 1,000 case batch
python scripts/pipeline/batch_trainer.py --batch_size 1000 --force

# Train with 5,000 case batch
python scripts/pipeline/batch_trainer.py --batch_size 5000 --force

# Monitor training
tail -f models/history.log
```

**Expected Performance:**

| Cases | Accuracy | F1 Weighted | Training Time |
|-------|----------|-------------|---------------|
| 1K    | 50-55%   | 40-45%      | ~10 min       |
| 5K    | 65-70%   | 55-60%      | ~25 min       |
| 10K   | 70-75%   | 60-65%      | ~40 min       |
| 20K   | 75-80%   | 65-70%      | ~1.5 hrs      |
| 50K   | 80-85%   | 70-75%      | ~3 hrs        |
| 100K  | 85-90%   | 75-80%      | ~5 hrs        |

### Training Configuration
- **Model:** Legal-BERT base uncased
- **Precision:** FP16 mixed precision
- **Batch Size:** 8 (adjustable)
- **Epochs:** 2
- **Format:** Safetensors (PyTorch 2.5.1+cu121)
- **Eval Strategy:** Steps (every 200 steps)

---

---

## 🌐 API Server

### Deploy on Render (Free tier, 512MB)

Set these environment variables in your Render service to keep memory within 512MB:

```
LOW_MEMORY=1
DISABLE_SEARCH_INDEX=1
DISABLE_SEMANTIC_INDEX=1
GUNICORN_WORKERS=1
GUNICORN_THREADS=1
```

Notes:
- With LOW_MEMORY enabled, only the classifier model is loaded; `/api/search` returns 503 unless you ship `models/search_index.pkl` and enable it.
- Gunicorn preloading is disabled and concurrency is minimal to avoid memory duplication.
- The Docker build excludes heavy folders (`models/production`, `models/runs`, `mlruns`, `data/raw`, `data/processed`) via `.dockerignore`.

### Start Server

```powershell
# Development server (FastAPI)
python run_server.py
# Server runs on http://0.0.0.0:5002

# Production server (Gunicorn)
gunicorn -c gunicorn.conf.py app:app
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/questions` | GET | Get questionnaire |
| `/api/questions/<case_type>` | GET | Questions for specific case type |
| `/api/analyze` | POST | Predict case outcome |
| `/api/search` | POST | Semantic search for similar cases |
| `/api/analyze_and_search` | POST | Combined prediction + search |
| `/version` | GET | API version & model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/api/drift/baseline` | GET | Baseline class distribution |
| `/api/drift/compare` | POST | Compare class distribution (drift detection) |

### Example Usage

```python
import requests

# Analyze case
response = requests.post('http://localhost:5002/api/analyze', json={
    'case_type': 'murder',
    'details': 'Case details here...'
})
prediction = response.json()

# Search similar cases
response = requests.post('http://localhost:5002/api/search', json={
    'query': 'murder with circumstantial evidence',
    'k': 5
})
similar_cases = response.json()
```

---

## 📖 Documentation

### Core Guides
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete project structure & metrics reference
- **[WORKSPACE_STATUS.md](WORKSPACE_STATUS.md)** - Current status, metrics, and recommendations
- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** - Training pipeline guide
- **[COLLECTOR_GUIDE.md](COLLECTOR_GUIDE.md)** - Data collection system guide

### Model Documentation
- **[MODEL_CARD.md](docs/MODEL_CARD.md)** - Model card with performance metrics
- **[DATA_SHEET.md](docs/DATA_SHEET.md)** - Dataset documentation

### Quick Commands

```powershell
# Status & Monitoring
python quick_status.py              # Quick stats
python metrics_dashboard.py         # Full dashboard

# Data Collection
python scripts/continuous_collector.py  # Start collector
python scripts/kanoon_harvest.py        # Legacy harvester

# Training
python scripts/pipeline/batch_trainer.py --batch_size 1000 --force
python scripts/train_model.py           # Legacy trainer

# API
python run_server.py                # Dev server
gunicorn -c gunicorn.conf.py app:app # Production

# Testing
python -m pytest -q                 # Run tests
python scripts/smoke_client.py      # API smoke test

# Database
sqlite3 data/legal_cases_10M.db "SELECT COUNT(*) FROM cases"
python scripts/build_search_index.py  # Build search index
```

---

## 🐳 Docker Deployment

### Docker Compose (Recommended)

```bash
docker compose up --build
```

### Manual Docker

```bash
# Build image
docker build -t ai-court .

# Run container
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/models/legal_case_classifier.pkl \
  ai-court
```

---

## 🧪 Testing

```powershell
# Run all tests
python -m pytest -q

# Run specific test
python -m pytest tests/test_api_basic.py -v

# Test with coverage
python -m pytest --cov=src --cov-report=html

# Smoke test API
python scripts/smoke_client.py
```

### Available Tests
- `test_api_basic.py` - Basic API functionality
- `test_drift_history.py` - Drift detection
- `test_model_metrics_endpoint.py` - Model metrics endpoint
- `test_regression_macro_f1.py` - F1 score regression
- `test_regression_relative.py` - Relative performance regression
- `test_version_and_metrics.py` - Version & metrics endpoints

---

## 📊 Monitoring & Metrics

### Real-Time Dashboard

```powershell
# Quick status (30 seconds)
python quick_status.py

# Comprehensive dashboard
python metrics_dashboard.py
```

**Dashboard Includes:**
- 📈 Data collection metrics (total cases, rates, distribution)
- 🤖 Model performance (accuracy, F1 scores, targets)
- ⚙️ System status (files, processes, health)
- 💡 Recommendations (next steps, training suggestions)

### Metrics Tracked

**Collection Metrics:**
- Total cases collected
- Collection rate (per hour/day)
- Court distribution
- Outcome distribution
- Temporal coverage (2015-2024)
- Database size & growth

**Model Metrics:**
- Accuracy (target: 70%+)
- F1 Weighted (target: 70%+)
- F1 Macro (target: 60%+)
- Precision & Recall per class
- Training time & device usage

**Data Quality:**
- Completeness score (78.2%)
- Text quality score (92.5%)
- Deduplication (100% unique via SHA256)
- Missing data analysis

---

## 🎯 Roadmap & Milestones

### Current Progress
- ✅ **1,000 cases** (Oct 5) - First milestone
- ✅ **5,000 cases** (Oct 6) - Second milestone
- ✅ **Model #1 trained** (60.24% accuracy on 413 cases)
- ⏳ **10,000 cases** (~28 min ETA) - Next milestone

### Upcoming Milestones

| Milestone | Cases | Expected Accuracy | ETA |
|-----------|-------|-------------------|-----|
| Model #2  | 10,000 | 70-75% | ~1 hour |
| Model #3  | 20,000 | 75-80% | ~14 hours |
| Production | 50,000 | 80-85% | ~44 hours |
| Enterprise | 100,000 | 85-90% | ~94 hours |

### Long-term Goals
- 🎯 100,000+ cases for enterprise-grade accuracy
- 🤖 Automated 5-hour training cycles
- 🌐 Production API deployment
- 📊 Real-time drift monitoring
- 🔍 Advanced semantic search with embeddings

---



This will build the image (injecting APP_VERSION/GIT_COMMIT if set), map port 8000, mount `./models` into the container, and expose `/api/health`, `/metrics`, and `/version`.

Environment variables:
- Set `APP_VERSION` and `GIT_COMMIT` at build time (Dockerfile build args) to surface values in `/version`.
- To protect endpoints, set `API_KEY` and pass header `X-API-Key: <value>`.
- For production rate limiting across replicas, use Redis and set `RATE_LIMIT_STORAGE_URI=redis://host:6379`.

## Model governance & run history

- Each training run is saved under `models/runs/<timestamp>_<shortuuid>/` with:
  - `legal_case_classifier.pkl`, `metrics.json`, `confusion_matrix.json`, `metadata.json`
- Latest run artifacts are copied to `models/` for API use.
- Lineage: `metadata.json` includes `run_id` and `previous_run`.
- History: `models/history.log` (JSON Lines) appends each run’s metadata.
- Regression guard: test suite checks for excessive accuracy drop between runs.
- Drift: `/api/drift/baseline` returns baseline class distribution and duplicate ratio; `/api/drift/compare` computes divergence from posted prediction histogram.
  - Agreement monitoring: `/api/metrics/agreement` exposes live classical vs multi-axis agreement stats (updated as requests come in).
  - Embedding / retrieval index drift: `scripts/drift_monitor.py --current retrieval_index/segments --previous <prev_dir> [--current-label-dist metadata.json --previous-label-dist prev_metadata.json]` outputs centroid shift, cosine distance, and KL/JS label distribution drift.
  - Store output as `drift_last.json` to have it included in consolidated `governance_status.json`.
  - Each compare call is appended (best-effort) to `logs/drift_history.log` for audit.
  - `/api/drift/history?limit=50` returns recent drift events (tail).
  - `/api/metrics/model` exposes latest evaluation metrics & selected metadata.
- Retrieval evaluation: multi-axis training auto-computes retrieval recall@K if `data/queries.csv` and `retrieval_index/segments` exist (`RETRIEVAL_EVAL_*` env vars) and saves under `retrieval_eval` in `metrics_multi_axis.json`.
- Gating: `scripts/model_gate.py` applies absolute + relative thresholds (macro F1, conflict rate, retrieval recall) to decide promotion → writes `models/multi_axis/promoted.json`.
- Shadow / primary multi-axis inference:
  - `ENABLE_MULTI_AXIS_SHADOW=1` (default) runs multi-axis model in parallel; `/api/analyze` returns classical + axis predictions + `agreement_rate`.
  - `USE_MULTI_AXIS_PRIMARY=1` promotes multi-axis relief/substantive/procedural fallback chain as the primary `judgment` in API responses (original classical label still available under `judgment_classical`).
- Agreement tracking: rolling agreement statistics maintained in-process; discrepancies sampled (future export to governance snapshot).
  - Persisted every 10 comparisons to `agreement_stats.json` (picked up by `governance_status.json`).
  - Governance endpoints: `GET /api/governance/status` (latest consolidated file), `POST /api/governance/refresh` (force rebuild).
  - Continuous refresh helper: `python scripts/refresh_governance_status.py 300` (interval seconds) to auto-update `governance_status.json`.
- Data quality: duplicate ratio and class distribution surfaced in metadata and endpoints.

- `src/ai_court/api/server.py` — Flask API
- `src/ai_court/model/legal_case_classifier.py` — training pipeline (TF-IDF + AdaBoost(RandomForest))
- `src/ai_court/scraper/legacy_kanoon.py` — Kanoon scraping (legacy)
- `src/ai_court/scraper/kanoon.py` — scraper wrapper module
- `src/ai_court/data/ingestion.py` — scraper wrapper → `data/raw/`
- `src/ai_court/data/prepare_dataset.py` — schema coercion → `data/processed/all_cases.csv`
- `scripts/build_search_index.py` — build semantic TF-IDF index → `models/search_index.pkl`
- `models/` — saved model artifacts
- `logs/` — scraper/debug logs
- `scripts/` — utilities (train_model.py, smoke_client.py, cleanup_repo.py)
- `docs/MODEL_CARD.md` — model description, metrics, limitations
- `docs/DATA_SHEET.md` — dataset description and considerations

## Notes

- Environment variables for governance:
  - `ALLOWED_ACCURACY_DROP` (default 0.15): max allowed relative drop in test accuracy between runs.
  - `DRIFT_JSD_WARN` (default 0.10): Jensen–Shannon divergence threshold for drift warning.
  - `DRIFT_JSD_ALERT` (default 0.20): threshold for drift alert.
    - `ALLOWED_ACCURACY_DROP` also enforced by tests; adjust as data scales.
    - `ALLOWED_MACRO_F1_DROP` (default 0.20) relative macro-F1 regression guard.

  CI / Automation variables (in GitHub Actions secrets recommended):
  - `API_KEY` (if you want CI smoke calls against protected endpoints in future extensions)
  - `SENTRY_DSN` (optional) for observability.

- Additional governance + multi-axis vars:
  - `MIN_MACRO_F1`, `MAX_CONFLICT_RATE`, `MIN_RETRIEVAL_RECALL`
  - `MACRO_F1_DROP_TOL`, `CONFLICT_RATE_INCREASE_TOL`, `RETRIEVAL_RECALL_DROP_TOL`
  - `RETRIEVAL_EVAL_QUERIES`, `RETRIEVAL_EVAL_K`, `RETRIEVAL_EVAL_MODEL`, `RETRIEVAL_EVAL_INDEX`
  - `ENABLE_MULTI_AXIS_SHADOW` (default 1), `USE_MULTI_AXIS_PRIMARY` (default 0), `MULTI_AXIS_INFER_MAX_LEN`
  - `SEM_RETRIEVAL_TOP_K`, `RETRIEVAL_TOP_K` (lexical/semantic context augmentation during training)

- Do not commit secrets. Use `.env` and environment variables.
- With tiny classes, stratified splitting is disabled to avoid errors.
- Add more judgments to improve accuracy; keep the schema.

## Production quickstart

- To check model drift:
  - `GET /api/drift/baseline` for reference distribution.
  - `POST /api/drift/compare` with `{ "counts": {"ClassA": 10, "ClassB": 5, ...} }` to compare live predictions.

1) Harvest data (optional at first):
  - Use default queries: `python scripts/kanoon_harvest.py`
  - Or provide your own: create `data/queries.csv` (see `data/queries.example.csv`), then
    - PowerShell:
     - `$env:KANOON_PAGES="10"; $env:KANOON_QUERIES_FILE="data/queries.csv"; python scripts/kanoon_harvest.py`

2) Build dataset + Train model:
  - Full pipeline: `python scripts/train_full_pipeline.py`
  - Skip harvest (use existing CSVs): `python scripts/train_full_pipeline.py --skip-harvest`
  - Evaluate saved model: `python scripts/evaluate_saved_model.py`

3) Serve API (dev):
  - `python run_server.py`

4) Serve API (prod via Docker):
  - `docker build -t ai-court .`
  - `docker run -p 8000:8000 -v %cd%/models:/app/models ai-court`
  - Or: `docker compose up --build`

5) Health check:
  - `GET /api/health` → `{ "status": "ok" }` when the model is ready
## Semantic search

- Build the index from scraped CSVs or the processed dataset fallback:
  - `python scripts/build_search_index.py`
- Build a semantic (dense) embedding index (SentenceTransformers):
  - `python scripts/build_semantic_index.py --model all-MiniLM-L6-v2`
  - Server prefers semantic index first if `SEMANTIC_INDEX_PATH` (default `models/semantic_index.pkl`) exists; falls back to TF-IDF.
- Query the API:
  - `POST /api/search` with `{ "query": "appeal dismissed conviction upheld", "k": 5 }`
- Test locally without server:
  - `python scripts/search_smoke.py`

Notes:
- The API auto-loads the search index from `SEARCH_INDEX_PATH` when present.
 - The API auto-loads the semantic embedding index from `SEMANTIC_INDEX_PATH` if present (higher relevance for longer queries).
- The server emits structured JSON logs with request IDs and latencies. Provide `X-Request-ID` to trace requests end-to-end.
 - The index builder prefers `data/raw_enriched/` automatically when available, otherwise falls back to `data/raw/` or the processed dataset.
 - For production rate limiting, set `RATE_LIMIT_STORAGE_URI=redis://<host>:<port>`; otherwise an in-memory backend is used (not recommended for multi-instance).
 - Optional error tracking: set `SENTRY_DSN` along with `SENTRY_TRACES_SAMPLE_RATE` and `SENTRY_PROFILES_SAMPLE_RATE` for performance telemetry.


## Scaling data & quality

- Target dataset sizes:
  - MVP: 5k–10k labeled judgments (≥500 per class)
  - Strong: 20k–50k (≥1k per class)
  - Enterprise: 100k–200k+
- Reduce the `Other` class by expanding `normalize_outcome()` mappings and auditing labels.
- Prefer temporal splits; monitor per-class F1 and confusion matrix.
