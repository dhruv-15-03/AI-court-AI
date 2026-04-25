"""Sync recorded case outcomes from the Java backend into the LabeledDataStore.

Pulls labeled cases from ``GET /api/cases/labeled-outcomes`` on the Java
service, appends each one to ``LabeledDataStore``, and (optionally) triggers
the ``RetrainingEngine`` when the configured retrain threshold is reached.

Usage:
    python scripts/sync_case_outcomes.py \\
        --java-url http://localhost:8080 \\
        --internal-key $INTERNAL_API_KEY

Env vars consumed as fallbacks:
    JAVA_BACKEND_URL       default http://localhost:8080
    INTERNAL_API_KEY       matched against Java's X-Internal-Key header
    RETRAIN_THRESHOLD      new-label count needed to auto-retrain (default 50)
    SYNC_STATE_FILE        path to timestamp checkpoint (default data/outcome_sync.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

# Ensure src/ is on path when run via `python scripts/...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_court.active_learning.loop import LabeledDataStore, RetrainingEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sync_case_outcomes")

DEFAULT_STATE = ROOT / "data" / "outcome_sync.json"
DEFAULT_LABELS = ROOT / "data" / "active_learning_labels.jsonl"


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"last_sync": 0, "synced_case_ids": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_sync": 0, "synced_case_ids": []}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def fetch_labeled_outcomes(java_url: str, internal_key: str | None, since: int) -> list[dict]:
    url = f"{java_url.rstrip('/')}/api/cases/labeled-outcomes"
    headers = {}
    if internal_key:
        headers["X-Internal-Key"] = internal_key
    resp = requests.get(url, headers=headers, params={"since": since}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body.get("cases", []) if isinstance(body, dict) else []


def sync(
    java_url: str,
    internal_key: str | None,
    labels_path: Path,
    state_path: Path,
    auto_retrain: bool,
    retrain_threshold: int,
) -> Dict[str, Any]:
    state = _load_state(state_path)
    synced_ids = set(state.get("synced_case_ids", []))

    logger.info("Fetching labeled outcomes since=%s", state.get("last_sync", 0))
    cases = fetch_labeled_outcomes(java_url, internal_key, int(state.get("last_sync", 0)))
    logger.info("Java returned %d labeled cases", len(cases))

    store = LabeledDataStore(store_path=str(labels_path))
    new_count = 0
    max_recorded = int(state.get("last_sync", 0))

    for c in cases:
        case_id = c.get("id")
        if case_id in synced_ids:
            continue
        text = (c.get("description") or "").strip()
        label = c.get("actualOutcome")
        if not text or not label:
            continue
        store.add_label(
            text=text,
            label=label,
            case_type=c.get("caseType") or "",
            source="court_outcome",
            metadata={
                "case_id": case_id,
                "title": c.get("title"),
                "predicted": c.get("predictedOutcome"),
                "notes": c.get("outcomeNotes"),
                "recorded_at": c.get("recordedAt"),
            },
        )
        synced_ids.add(case_id)
        new_count += 1
        if c.get("recordedAt") and int(c["recordedAt"]) > max_recorded:
            max_recorded = int(c["recordedAt"])

    state["last_sync"] = max_recorded or int(time.time())
    state["synced_case_ids"] = sorted(synced_ids)
    _save_state(state_path, state)

    result: Dict[str, Any] = {
        "fetched": len(cases),
        "new_labels": new_count,
        "total_labels": store.count(),
        "last_sync": state["last_sync"],
        "retrained": False,
    }

    if auto_retrain and new_count > 0:
        engine = RetrainingEngine(
            label_store=store,
            retrain_threshold=retrain_threshold,
            model_dir=str(ROOT / "models"),
        )
        if engine.should_retrain():
            logger.info("Retrain threshold reached — launching retraining cycle")
            try:
                meta = engine.retrain()
                result["retrained"] = True
                result["retrain_meta"] = {
                    "f1_score": meta.get("f1_score"),
                    "total_samples": meta.get("total_samples"),
                    "new_labels_used": meta.get("new_labels_used"),
                }
            except Exception as exc:  # pragma: no cover - heavy path
                logger.exception("Retrain failed: %s", exc)
                result["retrain_error"] = str(exc)
        else:
            logger.info("Retrain threshold not yet reached.")

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--java-url", default=os.getenv("JAVA_BACKEND_URL", "http://localhost:8080"))
    p.add_argument("--internal-key", default=os.getenv("INTERNAL_API_KEY"))
    p.add_argument("--labels-file", default=str(DEFAULT_LABELS))
    p.add_argument("--state-file", default=os.getenv("SYNC_STATE_FILE", str(DEFAULT_STATE)))
    p.add_argument("--no-retrain", action="store_true", help="Skip automatic retraining check")
    p.add_argument("--retrain-threshold", type=int,
                   default=int(os.getenv("RETRAIN_THRESHOLD", "50")))
    args = p.parse_args()

    result = sync(
        java_url=args.java_url,
        internal_key=args.internal_key,
        labels_path=Path(args.labels_file),
        state_path=Path(args.state_file),
        auto_retrain=not args.no_retrain,
        retrain_threshold=args.retrain_threshold,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
