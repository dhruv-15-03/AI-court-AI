"""Enhanced multi-axis transformer classifier with reconciliation & semantic retrieval.

Features:
 - Class weighting per axis
 - Optional lexical (TF-IDF) retrieval augmentation (append top-k similar docs)
 - Optional semantic retrieval augmentation (sentence-transformers cosine top-k)
 - Accuracy + macro-F1 per axis
 - Unified outcome reconciliation via `multi_axis_consistency.reconcile_axes`
 - Conflict rate & unified outcome distribution metrics
 - Run lineage directory structure under models/multi_axis/runs/<run_id>

Environment variables:
    MULTI_AXIS_MODEL          (default distilbert-base-uncased)
    RETRIEVAL_TOP_K           (int) lexical top-k (0 disables)
    RETRIEVAL_MAX_CORPUS      (int) limit corpus size for lexical index (default 5000)
    SEM_RETRIEVAL_TOP_K       (int) semantic top-k (0 disables)
    SEM_RETRIEVAL_MODEL       (default sentence-transformers/all-MiniLM-L6-v2)
    MULTI_AXIS_EPOCHS, MULTI_AXIS_BATCH, MULTI_AXIS_LR, MULTI_AXIS_MAX_LEN (used by pipeline shim)
"""
from __future__ import annotations
import os
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from datetime import datetime
import uuid

try:
    from .multi_axis_consistency import reconcile_axes  # type: ignore
except Exception:  # pragma: no cover
    def reconcile_axes(axis_preds: Dict[str,str]):  # fallback
        return {'unified_outcome': 'other', 'reason': {'buckets':{}, 'conflicts':[], 'precedence_order':[]}}

BACKBONE = os.getenv('MULTI_AXIS_MODEL','distilbert-base-uncased')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AXES = ['procedural_label','substantive_label','relief_label']

@dataclass
class Row:
    text: str
    labels: Dict[str,int]

class RetrievalAugmentor:
    def __init__(self, texts: List[str], top_k: int, max_corpus: int):
        self.top_k = top_k
        sample = texts[:max_corpus]
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(sample)
        self.corpus = sample
    def augment(self, text: str) -> str:
        if self.top_k <= 0:
            return text
        vec = self.vectorizer.transform([text])
        sims = (self.matrix @ vec.T).toarray().ravel()
        top_idx = sims.argsort()[::-1][:self.top_k]
        ctx = '\n'.join(self.corpus[i][:400] for i in top_idx if sims[i] > 0)
        return (text + '\n[CTX]\n' + ctx) if ctx else text

class AxisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, label_maps: Dict[str, Dict[str,int]], augmentor: RetrievalAugmentor | None):
        self.rows: List[Row] = []
        for _, r in df.iterrows():
            base_text = (str(r.get('case_type','')) + ' ' + str(r.get('case_data','')))[:9000]
            if augmentor:
                text = augmentor.augment(base_text)
            else:
                text = base_text
            labels = {}
            for ax in AXES:
                v = str(r.get(ax,'Unknown')) or 'Unknown'
                labels[ax] = label_maps[ax].setdefault(v, len(label_maps[ax]))
            self.rows.append(Row(text=text, labels=labels))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_maps = label_maps
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(row.text, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        for ax in AXES:
            item[ax] = torch.tensor(row.labels[ax], dtype=torch.long)
        return item

class MultiAxisModel(nn.Module):
    def __init__(self, backbone_name: str, num_labels: Dict[str,int]):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.heads = nn.ModuleDict({ax: nn.Linear(hidden, n) for ax,n in num_labels.items()})
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]
        return {ax: head(pooled) for ax, head in self.heads.items()}

@torch.no_grad()
def evaluate(model: MultiAxisModel, loader: DataLoader, axes=AXES):
    model.eval()
    preds_ax: Dict[str, List[int]] = {ax: [] for ax in axes}
    gold_ax: Dict[str, List[int]] = {ax: [] for ax in axes}
    for batch in loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        logits = model(input_ids, attention_mask)
        for ax in axes:
            pred = logits[ax].argmax(dim=1).cpu().tolist()
            preds_ax[ax].extend(pred)
            gold_ax[ax].extend(batch[ax].tolist())
    metrics = {}
    for ax in axes:
        correct = sum(int(a==b) for a,b in zip(preds_ax[ax], gold_ax[ax]))
        acc = correct/len(preds_ax[ax]) if preds_ax[ax] else 0.0
        # Macro-F1 over labels present
        try:
            macro = f1_score(gold_ax[ax], preds_ax[ax], average='macro')
        except Exception:
            macro = None
        metrics[ax] = {'accuracy': acc, 'macro_f1': macro}
    # Simple consistency: if substantive acquittal implies procedural allowed etc.
    inconsistencies = 0
    total = len(preds_ax['procedural_label'])
    for i in range(total):
        # Example rule: substantive_label class id of 'Acquittal' must align with procedural_label containing 'Appeal Allowed'
        # We can't decode text here without label maps; skip detailed semantics now.
        pass
    metrics['consistency_rules_checked'] = 0
    metrics['consistency_violations'] = inconsistencies
    return metrics

def compute_class_weights(label_maps: Dict[str, Dict[str,int]], dataset: AxisDataset) -> Dict[str, torch.Tensor]:
    counts = {ax: [0]*len(m) for ax,m in label_maps.items()}
    for row in dataset.rows:
        for ax,val in row.labels.items():
            counts[ax][val] += 1
    weights = {}
    for ax, arr in counts.items():
        import numpy as np
        arr_np = np.array(arr, dtype=float)
        arr_np[arr_np==0] = 1.0
        inv = 1.0/arr_np
        weights[ax] = torch.tensor(inv / inv.sum() * len(arr_np), dtype=torch.float)
    return weights

class SemanticRetriever:
    """Optional semantic retrieval using sentence-transformers embeddings."""
    def __init__(self, texts: List[str], model_name: str, top_k: int, max_corpus: Optional[int]=None):
        from sentence_transformers import SentenceTransformer
        if max_corpus is not None:
            texts = texts[:max_corpus]
        self.model = SentenceTransformer(model_name)
        self.texts = texts
        self.emb = self.model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        self.top_k = top_k
    def augment(self, text: str) -> str:
        if self.top_k <= 0:
            return text
        q = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        sims = (self.emb @ q.T).ravel()
        top = sims.argsort()[::-1][:self.top_k]
        ctx = '\n'.join(self.texts[i][:400] for i in top if sims[i] > 0.15)
        return text + ('\n[SEMCTX]\n' + ctx if ctx else '')

def _prepare_dataset(df: pd.DataFrame, tokenizer, args, label_maps):
    retrieval_k = int(os.getenv('RETRIEVAL_TOP_K','0'))
    retrieval_max = int(os.getenv('RETRIEVAL_MAX_CORPUS','5000'))
    augmentor_lex = None
    if retrieval_k > 0:
        print(f"[multi-axis] Building lexical retrieval index (top_k={retrieval_k})")
        corpus_texts = (df['case_type'].astype(str)+' '+df['case_data'].astype(str)).tolist()
        augmentor_lex = RetrievalAugmentor(corpus_texts, retrieval_k, retrieval_max)
    sem_top_k = int(os.getenv('SEM_RETRIEVAL_TOP_K','0'))
    sem_model = os.getenv('SEM_RETRIEVAL_MODEL','sentence-transformers/all-MiniLM-L6-v2')
    sem_retriever = None
    if sem_top_k > 0:
        try:
            print(f"[multi-axis] Building semantic retriever {sem_model} (top_k={sem_top_k})")
            corpus_texts = (df['case_type'].astype(str)+' '+df['case_data'].astype(str)).tolist()
            sem_retriever = SemanticRetriever(corpus_texts, sem_model, sem_top_k, None)
        except Exception as e:
            print(f"[multi-axis] Semantic retrieval disabled (error: {e})")
            sem_retriever = None

    class ComboAug:
        def augment(self, text: str):
            if augmentor_lex:
                text2 = augmentor_lex.augment(text)
            else:
                text2 = text
            if sem_retriever:
                text2 = sem_retriever.augment(text2)
            return text2
    combo = ComboAug()
    dataset = AxisDataset(df, tokenizer, args.max_len, label_maps, combo if (augmentor_lex or sem_retriever) else None)
    return dataset, {'lex_top_k': retrieval_k, 'sem_top_k': sem_top_k, 'sem_model': sem_model if sem_top_k>0 else None}

def train(args):
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    df = df[df['case_data'].astype(str).str.strip().astype(bool)]
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    label_maps: Dict[str, Dict[str,int]] = {ax: {} for ax in AXES}
    dataset, retrieval_meta = _prepare_dataset(df, tokenizer, args, label_maps)
    num_labels = {ax: len(m) for ax,m in label_maps.items()}
    model = MultiAxisModel(BACKBONE, num_labels).to(DEVICE)
    class_weights = compute_class_weights(label_maps, dataset)
    criterion = {ax: nn.CrossEntropyLoss(weight=class_weights[ax].to(DEVICE)) for ax in AXES}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    split = int(0.85 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            loss = 0
            for ax in AXES:
                loss += criterion[ax](logits[ax], batch[ax].to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_metrics = evaluate(model, val_loader)
        summary = ' '.join([f"{ax[:4]}Acc={val_metrics[ax]['accuracy']:.3f}/F1={(val_metrics[ax]['macro_f1'] or 0):.3f}" for ax in AXES])
        print(f"Epoch {epoch+1} loss={total_loss/len(train_loader):.4f} | {summary}")
    # Collect reconciliation metrics on validation set
    inv_maps = {ax: {v:k for k,v in mp.items()} for ax, mp in label_maps.items()}
    recon_samples = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            batch_size = ids.size(0)
            for i in range(batch_size):
                axis_pred_labels = {}
                for ax in AXES:
                    pred_idx = int(logits[ax][i].argmax().item())
                    axis_pred_labels[ax] = inv_maps[ax].get(pred_idx, 'Unknown')
                rec = reconcile_axes(axis_pred_labels)
                recon_samples.append({'axes': axis_pred_labels, 'unified': rec['unified_outcome'], 'reason': rec['reason']})
    from collections import Counter
    unified_counts = Counter([r['unified'] for r in recon_samples])
    axis_label_counts = {ax: Counter([r['axes'][ax] for r in recon_samples]) for ax in AXES}
    # Conflict: any non-primary bucket in reason.conflicts
    conflict_rows = sum(1 for r in recon_samples if r['reason'].get('conflicts'))
    conflict_rate = conflict_rows / len(recon_samples) if recon_samples else 0.0

    metrics = evaluate(model, val_loader)
    macro_vals = [m['macro_f1'] for m in metrics.values() if isinstance(m, dict) and m.get('macro_f1') is not None]
    macro_avg = sum(macro_vals)/len(macro_vals) if macro_vals else None
    reconciliation = {
        'unified_distribution': dict(unified_counts),
        'axis_label_counts': {ax: dict(cnt) for ax,cnt in axis_label_counts.items()},
        'conflict_rate': conflict_rate,
        'num_validation_samples': len(recon_samples),
        'sample_conflicts': [r for r in recon_samples if r['reason'].get('conflicts')][:5],
        'macro_f1_axes_avg': macro_avg,
        'retrieval': retrieval_meta,
    }
    run_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:8]
    base_dir = 'models/multi_axis'
    run_dir = os.path.join(base_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    artifact_bundle = {'model_state': model.state_dict(), 'label_maps': label_maps, 'backbone': BACKBONE, 'run_id': run_id}
    torch.save(artifact_bundle, os.path.join(run_dir,'multi_axis.pt'))
    with open(os.path.join(run_dir,'label_maps.json'),'w',encoding='utf-8') as f:
        json.dump({ax: {k:v for k,v in m.items()} for ax,m in label_maps.items()}, f, indent=2)
    # Attempt to link retrieval index metadata if exists
    retrieval_index_meta = None
    try:
        idx_meta_path = os.path.join('retrieval_index','segments','index_meta.json')
        if os.path.exists(idx_meta_path):
            with open(idx_meta_path,'r',encoding='utf-8') as rf:
                retrieval_index_meta = json.load(rf)
    except Exception as _re:
        retrieval_index_meta = {'error': str(_re)}
    # Optional retrieval recall evaluation (integrate evaluate_retrieval style logic)
    retrieval_eval = None
    try:
        queries_csv = os.getenv('RETRIEVAL_EVAL_QUERIES','data/queries.csv')
        eval_k = int(os.getenv('RETRIEVAL_EVAL_K','10'))
        eval_model = os.getenv('RETRIEVAL_EVAL_MODEL','sentence-transformers/all-MiniLM-L6-v2')
        index_dir = os.getenv('RETRIEVAL_EVAL_INDEX','retrieval_index/segments')
        if os.path.exists(queries_csv) and os.path.exists(os.path.join(index_dir,'embeddings.npy')):
            import numpy as _np
            import pandas as _pd
            import json as _json
            from sentence_transformers import SentenceTransformer as _ST
            emb = _np.load(os.path.join(index_dir,'embeddings.npy'))
            metas = []
            seg_path = os.path.join(index_dir,'segments.jsonl')
            if os.path.exists(seg_path):
                with open(seg_path,'r',encoding='utf-8') as sf:
                    for line in sf:
                        try:
                            metas.append(_json.loads(line))
                        except Exception:
                            pass
            if metas:
                model_rt = _ST(eval_model)
                norms = _np.linalg.norm(emb, axis=1, keepdims=True)+1e-12
                emb_n = emb / norms
                qdf = _pd.read_csv(queries_csv)
                hits = 0
                results = []
                for _, row in qdf.iterrows():
                    q = str(row['query'])
                    expected = str(row['expected_substring']).lower()
                    q_vec = model_rt.encode([q], convert_to_numpy=True, normalize_embeddings=True)
                    sims = (q_vec @ emb_n.T).ravel()
                    top_idx = sims.argsort()[::-1][:eval_k]
                    found = any(expected in (metas[i].get('text','') or '').lower() for i in top_idx)
                    hits += 1 if found else 0
                    results.append({'query': q, 'found': found, 'expected': expected, 'top_ids': [metas[i].get('segment_id') for i in top_idx]})
                recall = hits / len(qdf) if len(qdf) else 0.0
                retrieval_eval = {'k': eval_k, 'recall_at_k': recall, 'results': results, 'queries': len(qdf)}
                print(f"[multi-axis] retrieval recall@{eval_k}={recall:.3f} over {len(qdf)} queries")
    except Exception as _reval:
        retrieval_eval = {'error': str(_reval)}

    full_metrics = {
        'axes': metrics,
        'reconciliation': reconciliation,
        'run_id': run_id,
        'backbone': BACKBONE,
        'retrieval_index': retrieval_index_meta,
        'retrieval_eval': retrieval_eval
    }
    with open(os.path.join(run_dir,'metrics_multi_axis.json'),'w',encoding='utf-8') as f:
        json.dump(full_metrics, f, indent=2)
    # Update latest copies
    for name in ['multi_axis.pt','label_maps.json','metrics_multi_axis.json']:
        import shutil
        shutil.copy2(os.path.join(run_dir, name), os.path.join(base_dir, name))
    # Append history
    try:
        with open(os.path.join(base_dir,'multi_axis_history.log'),'a',encoding='utf-8') as hf:
            hf.write(json.dumps({'run_id': run_id, 'conflict_rate': conflict_rate, 'macro_f1_axes_avg': macro_avg, 'retrieval': retrieval_meta})+'\n')
    except Exception:
        pass
    print(f"[multi-axis] Saved model + metrics -> {run_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--max_len', type=int, default=512)
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
