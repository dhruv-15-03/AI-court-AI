"""Consistency reconciliation for multi-axis predictions.

Rules (initial draft):
  - If procedural_label indicates dismissal and substantive_label suggests acquittal/overturn, prefer acquittal composite outcome.
  - Relief/Disposition synthesis: Combine axes into a synthetic unified_outcome with precedence order.

This module provides:
  reconcile_axes(axis_preds: dict[str,str]) -> dict with unified_outcome + reasons

Future extensions:
  - Data-driven conditional probability table.
  - Conflict confidence scoring.
"""
from __future__ import annotations
from typing import Dict, Any

# Simple keyword sets (placeholder heuristics)
ACQUITTAL_KEYS = {"acquittal", "acquitted", "overturned"}
DISMISS_KEYS = {"dismiss", "dismissed"}
GRANT_KEYS = {"granted", "allowed"}
DENY_KEYS = {"denied", "refused", "rejected"}

PRECEDENCE = ["acquittal", "grant", "dismiss", "deny", "other"]

def _bucket(label: str) -> str:
    l = (label or "").lower()
    if any(k in l for k in ACQUITTAL_KEYS): return "acquittal"
    if any(k in l for k in GRANT_KEYS): return "grant"
    if any(k in l for k in DISMISS_KEYS): return "dismiss"
    if any(k in l for k in DENY_KEYS): return "deny"
    return "other"


def reconcile_axes(axis_preds: Dict[str,str]) -> Dict[str, Any]:
    buckets = {ax: _bucket(lbl) for ax,lbl in axis_preds.items()}
    # Precedence driven resolution
    for tag in PRECEDENCE:
        if tag in buckets.values():
            unified = tag
            break
    else:
        unified = "other"
    # Reasons: list conflicting axes
    conflicts = []
    primary = unified
    for ax, tag in buckets.items():
        if tag != primary and tag != 'other':
            conflicts.append(ax)
    reason = {
        'buckets': buckets,
        'conflicts': conflicts,
        'precedence_order': PRECEDENCE
    }
    return {
        'unified_outcome': unified,
        'reason': reason
    }

if __name__ == '__main__':
    sample = { 'procedural_label': 'Appeal Dismissed', 'substantive_label': 'Acquittal Granted', 'relief_label': 'Bail Denied'}
    print(reconcile_axes(sample))
