# Professional Pipeline Guide

## 🎯 Overview

This is the **professional batch training pipeline** for AI Court. It runs in automated 5-hour cycles to continuously improve the model.

---

## 📁 Pipeline Structure

```
scripts/pipeline/
├── batch_trainer.py          # Core batch training engine
├── run_5hour_cycle.py        # 5-hour cycle orchestrator
├── monitor_pipeline.py       # Real-time monitoring dashboard
├── generate_embeddings.py    # (Coming soon)
└── build_index.py           # (Coming soon)
```

---

## 🚀 Quick Start

### Option 1: Run Single Training Batch

Train on current cases (minimum 500 cases):

```bash
python scripts/pipeline/batch_trainer.py --batch_size 500
```

Force training even with fewer cases:

```bash
python scripts/pipeline/batch_trainer.py --force
```

### Option 2: Run Full 5-Hour Cycle

Automated cycle (collect + train + embed + index):

```bash
python scripts/pipeline/run_5hour_cycle.py
```

This will:
1. **Collect** data (2.5 hours) - Harvest ~3,000-6,000 cases
2. **Train** model (1 hour) - Batch training every 500+ cases
3. **Embeddings** (1 hour) - Generate 768-dim vectors
4. **Index** (30 min) - Build Faiss search index

### Option 3: Monitor Progress

Real-time dashboard:

```bash
python scripts/pipeline/monitor_pipeline.py
```

Shows:
- Total cases collected
- Collection rate (cases/hour)
- Training metrics (accuracy, F1)
- Milestones (1k, 10k, 100k)
- Cycle status

---

## 📊 How It Works

### Batch Training System

The `batch_trainer.py` is the core engine:

1. **Check Database**: Count total cases
2. **Check Last Training**: How many cases in last model?
3. **Calculate New Cases**: `total - last_trained`
4. **Decide**: Train if `new_cases >= batch_size`
5. **Train**: Fine-tune Legal-BERT on GPU
6. **Save**: Production model + metadata

**Key Features:**
- ✅ Incremental learning (trains on ALL cases each time, not just new ones)
- ✅ Auto-resume from checkpoints
- ✅ GPU optimization (RTX 4050)
- ✅ Mixed precision (FP16)
- ✅ Memory efficient (small batches)

### 5-Hour Cycle Orchestrator

The `run_5hour_cycle.py` manages the full workflow:

```
Cycle Start
    ↓
Phase 1: Collection (2.5h)
    → Harvester runs in background
    → Target: 3,000-6,000 cases
    ↓
Phase 2: Training (1h)
    → Wait 30 min for cases
    → Train batch (500+ cases)
    → Save production model
    ↓
Phase 3: Embeddings (1h)
    → Generate vectors
    → Save to data/embeddings/
    ↓
Phase 4: Indexing (30m)
    → Build Faiss index
    → Ready for search
    ↓
Cycle Complete
    ↓
Wait 5 minutes
    ↓
Next Cycle...
```

---

## 📈 Model Progression

As you collect more data, the model improves:

| Cases | Expected Accuracy | Expected F1 | Status |
|-------|------------------|-------------|--------|
| 100-500 | 55-60% | 0.50-0.55 | POC |
| 500-1k | 60-65% | 0.55-0.60 | Alpha |
| 1k-10k | 65-75% | 0.60-0.70 | Beta |
| 10k-100k | 75-85% | 0.70-0.80 | V1.0 |
| 100k+ | 85-90%+ | 0.80-0.85+ | Production |

Each 5-hour cycle improves the model incrementally.

---

## 🔧 Configuration

### Batch Trainer Settings

Edit `batch_trainer.py`:

```python
BATCH_SIZE = 8              # GPU batch size (6GB VRAM)
GRADIENT_ACCUMULATION = 4   # Effective batch = 32
EPOCHS_PER_BATCH = 2        # Epochs per training
LEARNING_RATE = 2e-5        # Learning rate
FP16 = True                 # Mixed precision (2x speed)
```

### Minimum Cases for Training

Default: 1,000 cases

Change with flag:
```bash
python batch_trainer.py --batch_size 500  # Lower threshold
```

---

## 📂 Output Files

### Production Models

Saved to `models/production/`:

```
models/production/
├── model_20250106_112345/    # Latest model
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
├── model_20250106_102345/    # Previous model
└── training_metadata.json    # Metadata (accuracy, F1, etc.)
```

### Checkpoints

Saved to `models/checkpoints/`:

```
models/checkpoints/
├── batch_20250106_112345/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-1500/
```

Resume from checkpoint if training crashes.

### Logs

Saved to `logs/`:

```
logs/
├── batch_trainer.log         # Training logs
├── pipeline_5h.log           # Cycle logs
├── cycle_1_20250106_110000.json  # Cycle summary
└── cycle_2_20250106_160000.json
```

---

## 🎯 Best Practices

### 1. Monitor Progress

Always run the monitor in a separate terminal:

```bash
# Terminal 1: Run pipeline
python scripts/pipeline/run_5hour_cycle.py

# Terminal 2: Monitor
python scripts/pipeline/monitor_pipeline.py
```

### 2. Check Logs

If something fails, check logs:

```bash
# Latest training log
type logs\batch_trainer.log

# Latest cycle log
type logs\pipeline_5h.log
```

### 3. GPU Monitoring

Watch GPU usage:

```bash
nvidia-smi -l 1
```

Should see:
- GPU: RTX 4050
- Memory: ~4-5 GB used during training
- Utilization: 90-100%

### 4. Validate Models

After training, check metadata:

```bash
type models\production\training_metadata.json
```

Look for:
- `accuracy` > 0.60 (good)
- `f1_macro` > 0.55 (good)
- Improving over time

---

## 🐛 Troubleshooting

### "Not enough cases to train"

**Problem:** < 100 cases in database

**Solution:** Wait for harvester to collect more, or use `--force`:
```bash
python batch_trainer.py --force
```

### "CUDA out of memory"

**Problem:** GPU VRAM exceeded

**Solution:** Reduce batch size in `batch_trainer.py`:
```python
BATCH_SIZE = 4  # Reduce from 8
```

### "Database locked"

**Problem:** Multiple processes accessing database

**Solution:** Stop harvester, then train:
```bash
# Stop harvester (Ctrl+C in that terminal)
# Then run trainer
python batch_trainer.py
```

### Training too slow

**Problem:** Each epoch takes > 30 minutes

**Solution:** 
1. Check GPU is being used: `nvidia-smi`
2. Verify FP16 enabled: `FP16 = True` in code
3. Reduce data: Train on 80% of cases, not 100%

---

## 🔄 Continuous Operation

For 24/7 operation:

1. **Start harvester** (runs forever):
   ```bash
   python scripts/ultra_fast_harvester.py
   ```

2. **Start pipeline** (5h cycles):
   ```bash
   python scripts/pipeline/run_5hour_cycle.py
   ```

3. **Monitor** (optional):
   ```bash
   python scripts/pipeline/monitor_pipeline.py
   ```

Let it run for days/weeks to reach 100k+ cases.

---

## 📊 Expected Timeline

At 600 cases/hour collection rate:

| Target | Cases | Days | Expected Accuracy |
|--------|-------|------|------------------|
| Alpha | 1,000 | 2 | 60-65% |
| Beta | 10,000 | 17 | 70-75% |
| V1.0 | 100,000 | 167 | 80-85% |
| V2.0 | 1,000,000 | 1,667 | 85-90% |

With 5-hour cycles, you get:
- ~5 training updates per day
- Continuous improvement
- Auto-checkpointing

---

## 🎉 Next Steps

1. **Start collecting**: Harvester is already running!
2. **Wait for 500 cases**: Check with `monitor_pipeline.py`
3. **Run first batch**: `python batch_trainer.py --batch_size 500 --force`
4. **Validate results**: Check `models/production/training_metadata.json`
5. **Start 5h cycles**: `python run_5hour_cycle.py`
6. **Let it run**: 24/7 for best results

---

## 📞 Quick Commands Reference

```bash
# Monitor progress
python scripts/pipeline/monitor_pipeline.py

# Train now (force)
python scripts/pipeline/batch_trainer.py --force

# Run 5h cycle
python scripts/pipeline/run_5hour_cycle.py

# Check database
python scripts/check_progress.py

# GPU status
nvidia-smi
```

---

**You're all set! Start the pipeline and watch the magic happen! 🚀**
