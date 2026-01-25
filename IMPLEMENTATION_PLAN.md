# 🚀 AI-Court Implementation Plan v2.0

> **Goal**: Optimize AI-Court for free-tier deployment (Render 512MB) while adding all suggested improvements at zero cost.

---

## 📊 Current State Analysis

### System Metrics
| Metric | Current Value | Target |
|--------|---------------|--------|
| Memory Usage | ~400MB | < 450MB |
| Model Size | ~15MB (TF-IDF + RF) | Same |
| Test Accuracy | 84.5% | ≥ 85% |
| Minority Class F1 | 0.636 | ≥ 0.70 |
| API Response Time | ~200ms | < 300ms |

### Deployment Constraints (Render Free Tier)
- **RAM**: 512MB hard limit
- **CPU**: Shared, burst-limited
- **Disk**: 1GB (sufficient for our model)
- **Cold Start**: ~30s (acceptable)
- **Spin Down**: After 15 min inactivity

---

## 🎯 Implementation Phases

## Phase 1: Quick Wins (Day 1) ✅ Priority
> Focus: Immediate value, no architecture changes

### 1.1 Fix Preprocessor Warning
**File**: `src/ai_court/model/preprocessor.py`
- Add proper else clause at lines 26-27
- Improve logging messages

### 1.2 Confidence-Based Abstention
**File**: `src/ai_court/api/routes/analysis.py`
- Add configurable `CONFIDENCE_THRESHOLD` (default: 0.5)
- Return `needs_review: true` for low-confidence predictions
- Auto-queue uncertain predictions to Active Learning

### 1.3 Wire Active Learning Queue
**Files**: `src/ai_court/api/routes/analysis.py`, `src/ai_court/api/routes/feedback.py`
- Automatically add low-confidence predictions to AL queue
- Add reason for queuing (uncertainty, abstention, user flag)
- Persist queue to disk

### 1.4 Add Prediction Explainability
**File**: `src/ai_court/api/routes/analysis.py`
- Extract top-K TF-IDF features contributing to prediction
- Return `key_factors: []` in API response
- Memory-efficient implementation (no extra model load)

---

## Phase 2: Data Quality (Day 2) 
> Focus: Remove external dependencies, improve training

### 2.1 Local Extractive Summarization
**File**: `src/ai_court/scraper/kanoon.py`
- Replace HuggingFace API calls with local extraction
- Improve `extract_judgment_section()` with more patterns
- Add fallback summary from intro + conclusion

### 2.2 Class Imbalance Handling
**File**: `src/ai_court/model/trainer.py`
- Add optional SMOTE/ADASYN for minority oversampling
- Use `imbalanced-learn` (already lightweight)
- Make it configurable via env var

### 2.3 Stratified Sampling Improvements
**File**: `src/ai_court/data/loader.py`
- Ensure stratified train/test split (already present, verify)
- Add minimum samples guard for minority classes

---

## Phase 3: RAG & Search (Day 3)
> Focus: Complete placeholder functionality

### 3.1 Lightweight Semantic Search (Optional)
**File**: `scripts/build_semantic_index.py`
- Use `all-MiniLM-L6-v2` (~22MB, runs on CPU)
- Batch encoding with memory limits
- Disable by default for free tier

### 3.2 Complete RAG Pipeline
**File**: `src/ai_court/rag/pipeline.py`
- Implement `retrieve()` using existing TF-IDF index
- Format context from retrieved docs
- No LLM generation (retrieval-only mode)

### 3.3 Hybrid Search Improvements
**File**: `src/ai_court/retrieval/hybrid.py`
- Add weighted fusion option
- Add outcome filtering for search results

---

## Phase 4: Deployment Optimization (Day 4)
> Focus: Memory, performance, production readiness

### 4.1 Memory Optimization
**Files**: `src/ai_court/api/dependencies.py`, `gunicorn.conf.py`
- Add explicit memory guards
- Lazy loading for optional features
- Clear caches on low memory

### 4.2 Environment Configuration
**File**: `.env.example`
- Document all new env vars
- Add sensible defaults for free tier

### 4.3 Health Check Improvements
**File**: `src/ai_court/api/routes/monitoring.py`
- Add memory usage endpoint
- Add readiness vs liveness probes

---

## 📁 Files to Create/Modify

### New Files
| File | Purpose |
|------|---------|
| `src/ai_court/utils/explainability.py` | Feature importance extraction |
| `src/ai_court/scraper/extractive_summary.py` | Local summarization utils |

### Modified Files
| File | Changes |
|------|---------|
| `src/ai_court/model/preprocessor.py` | Fix else branch |
| `src/ai_court/api/routes/analysis.py` | Confidence threshold, AL queue, explainability |
| `src/ai_court/api/routes/feedback.py` | Auto-queue support |
| `src/ai_court/api/state.py` | New state vars |
| `src/ai_court/api/config.py` | New config vars |
| `src/ai_court/scraper/kanoon.py` | Local summary |
| `src/ai_court/model/trainer.py` | Imbalance handling |
| `src/ai_court/rag/pipeline.py` | Complete implementation |
| `requirements.txt` | Add imbalanced-learn |

---

## 🔧 Configuration Variables

### New Environment Variables
```bash
# Confidence & Abstention
CONFIDENCE_THRESHOLD=0.5           # Predictions below this get flagged
AUTO_QUEUE_LOW_CONFIDENCE=1        # Auto-add to AL queue

# Explainability
EXPLAIN_TOP_K=5                    # Number of key factors to return

# Memory Optimization
MAX_SEARCH_RESULTS=10              # Limit search results
LAZY_LOAD_SEARCH=1                 # Load search index on first request

# Summarization
USE_LOCAL_SUMMARY=1                # Use local extractive (no HF API)

# Imbalance Handling
ENABLE_SMOTE=0                     # Enable during training only
SMOTE_SAMPLING_STRATEGY=0.5        # Target ratio for minority classes
```

---

## 💾 Memory Budget (512MB Target)

| Component | Memory | Status |
|-----------|--------|--------|
| Python Runtime | ~50MB | Required |
| Flask + Dependencies | ~80MB | Required |
| TF-IDF Model | ~15MB | Required |
| RandomForest | ~100MB | Required |
| NLTK Data | ~30MB | Required |
| Search Index | ~50MB | Optional (lazy) |
| Buffer/Headroom | ~187MB | Safety margin |
| **Total** | **~325-512MB** | ✅ Within limit |

---

## 🧪 Testing Checklist

### Phase 1 Tests
- [ ] Preprocessor loads rules correctly
- [ ] Confidence threshold triggers abstention
- [ ] AL queue persists and loads
- [ ] Explainability returns valid factors

### Phase 2 Tests
- [ ] Local summary extracts key sections
- [ ] SMOTE improves minority class metrics (offline)
- [ ] No HF API calls during harvesting

### Phase 3 Tests
- [ ] RAG returns relevant documents
- [ ] Hybrid search ranks correctly

### Phase 4 Tests
- [ ] Memory stays under 450MB under load
- [ ] Health checks pass consistently
- [ ] Cold start completes in < 60s

---

## 🚀 Deployment Checklist

### Before Deploy
- [ ] Run `python -m pytest -q`
- [ ] Check memory with `python -c "import tracemalloc; ..."`
- [ ] Verify `.env` has correct values
- [ ] Build Docker image locally

### Deploy Commands (Render)
```bash
# Build command (Render)
pip install -r requirements.txt

# Start command (Render)  
gunicorn -c gunicorn.conf.py src.ai_court.api.server:app

# Environment variables (set in Render dashboard)
LOW_MEMORY=1
DISABLE_SEARCH_INDEX=0
DISABLE_SEMANTIC_INDEX=1
GUNICORN_WORKERS=1
GUNICORN_THREADS=2
CONFIDENCE_THRESHOLD=0.5
AUTO_QUEUE_LOW_CONFIDENCE=1
USE_LOCAL_SUMMARY=1
```

---

## 📈 Success Metrics

| Metric | Before | After | Method |
|--------|--------|-------|--------|
| Memory (prod) | ~400MB | < 450MB | Render metrics |
| Minority F1 | 0.636 | ≥ 0.70 | `models/metrics.json` |
| API Latency P95 | ~200ms | < 300ms | Prometheus |
| Low-conf rate | Unknown | < 15% | AL queue stats |
| HF API calls | N/harvest | 0 | Logs |

---

## 🛡️ Rollback Plan

If issues arise after deployment:

1. **Revert Docker image** to previous tag
2. **Disable new features** via env vars:
   ```bash
   AUTO_QUEUE_LOW_CONFIDENCE=0
   CONFIDENCE_THRESHOLD=0.0  # Disable abstention
   ```
3. **Check logs** in Render dashboard
4. **Run health check**: `curl https://your-app.onrender.com/api/health`

---

## 📝 Implementation Order

```
Day 1 (Quick Wins):
  1.1 → 1.2 → 1.3 → 1.4 (sequential, each builds on previous)

Day 2 (Data Quality):  
  2.1 → 2.2 → 2.3 (can be parallel)

Day 3 (RAG & Search):
  3.2 → 3.3 (3.1 optional, skip for free tier)

Day 4 (Optimization):
  4.1 → 4.2 → 4.3 → Deploy
```

---

*Plan created: 2026-01-24*
*Target completion: 2026-01-28*
