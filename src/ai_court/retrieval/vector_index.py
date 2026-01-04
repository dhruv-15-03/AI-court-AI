"""Lightweight vector index utilities (numpy cosine) with optional FAISS.

Stores embeddings + metadata; supports top-k similarity search.
"""
from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

try:  # optional
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()

@dataclass
class VectorIndex:
    embeddings: np.ndarray
    meta: List[Dict[str, Any]]
    use_faiss: bool = False

    def __post_init__(self):
        if self.use_faiss and _HAS_FAISS:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            # Normalize for cosine
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            # Pre-normalize for cosine
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12
            self.embeddings = self.embeddings / norms
            self.index = None

    def search(self, query_vecs: np.ndarray, k: int = 5) -> List[List[Tuple[int, float]]]:
        if self.use_faiss and _HAS_FAISS and self.index is not None:
            q = query_vecs.copy()
            faiss.normalize_L2(q)
            sims, idxs = self.index.search(q, k)
            results = []
            for row_idx in range(q.shape[0]):
                results.append([(int(idxs[row_idx, j]), float(sims[row_idx, j])) for j in range(k) if idxs[row_idx, j] != -1])
            return results
        # Numpy fallback
        q = query_vecs
        norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        qn = q / norms
        sims = qn @ self.embeddings.T
        results = []
        for i in range(qn.shape[0]):
            top_idx = np.argsort(-sims[i])[:k]
            results.append([(int(j), float(sims[i, j])) for j in top_idx])
        return results

    def save(self, out_dir: str, model_name: str):
        os.makedirs(out_dir, exist_ok=True)
        emb_path = os.path.join(out_dir, 'embeddings.npy')
        np.save(emb_path, self.embeddings.astype('float32'))
        meta_path = os.path.join(out_dir, 'segments.jsonl')
        with open(meta_path, 'w', encoding='utf-8') as f:
            for m in self.meta:
                f.write(json.dumps(m) + '\n')
        # Hash
        with open(emb_path, 'rb') as ef:
            emb_hash = sha256_bytes(ef.read())
        # Centroid statistics (useful for drift detection)
        centroid = self.embeddings.mean(axis=0)
        mean_norm = float(np.linalg.norm(centroid))
        centroid_hash = sha256_bytes(centroid.tobytes())
        # Persist centroid vector for precise drift computations
        centroid_path = os.path.join(out_dir, 'centroid.npy')
        try:
            np.save(centroid_path, centroid.astype('float32'))
        except Exception:
            centroid_path = None
        meta_json = {
            'model': model_name,
            'dimension': int(self.embeddings.shape[1]),
            'count': int(self.embeddings.shape[0]),
            'embeddings_hash': emb_hash,
            'centroid_hash': centroid_hash,
            'centroid_norm': mean_norm,
            'centroid_path': 'centroid.npy' if centroid_path else None,
            'faiss': self.use_faiss and _HAS_FAISS
        }
        with open(os.path.join(out_dir, 'index_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_json, f, indent=2)
        return meta_json

    @classmethod
    def load(cls, index_dir: str):
        import numpy as np
        emb = np.load(os.path.join(index_dir, 'embeddings.npy'))
        meta = []
        with open(os.path.join(index_dir, 'segments.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    meta.append(json.loads(line))
                except Exception:
                    pass
        return cls(embeddings=emb, meta=meta, use_faiss=False)
