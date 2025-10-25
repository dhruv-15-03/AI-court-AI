"""Hierarchical Classification Model Scaffold (Phase 3).

Design:
 - Single shared encoder (e.g., Legal-BERT or domain-adapted encoder)
 - Multiple classification heads: Level 1 (coarse), Level 2 (leaf)
 - Support multi-label (sigmoid heads) in future.

Current status: placeholder forward interface & config dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:  # optional heavy dep
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


@dataclass
class HierConfig:
    encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    level1_classes: int = 8
    level2_classes: int = 32
    dropout: float = 0.1


class HierarchicalOutcomeModel(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: HierConfig):  # type: ignore[override]
        if torch is None:
            raise RuntimeError('PyTorch not installed; install torch to use hierarchical model.')
        super().__init__()
        from transformers import AutoModel
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.level1_head = nn.Linear(hidden, cfg.level1_classes)
        self.level2_head = nn.Linear(hidden, cfg.level2_classes)

    def forward(self, input_ids, attention_mask):  # type: ignore[override]
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = enc.last_hidden_state[:,0]
        x = self.dropout(pooled)
        return {
            'level1_logits': self.level1_head(x),
            'level2_logits': self.level2_head(x)
        }


__all__ = ['HierConfig','HierarchicalOutcomeModel']
