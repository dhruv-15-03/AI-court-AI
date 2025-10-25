"""
Evaluate the saved model (models/legal_case_classifier.pkl) against a CSV dataset.
The CSV must map to columns: ['case_data','case_type','judgement'] (prepare_dataset can help).
"""
import os
import sys
import json
import dill
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ai_court.api.server import MODEL_PATH  # noqa: E402 - reuse path logic


def evaluate(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    with open(MODEL_PATH, "rb") as f:
        saved = dill.load(f)
    model = saved["model"]
    label_encoder = saved["label_encoder"]
    preprocess_fn = saved.get("preprocessor")

    df = pd.read_csv(csv_path)
    required = ["case_data", "case_type", "judgement"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV needs columns: {required}")

    texts = (df["case_type"].astype(str).str.lower() + " " + df["case_data"].astype(str)).tolist()
    X = [preprocess_fn(t) for t in texts]
    y_true = label_encoder.transform(df["judgement"].astype(str))

    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    present = sorted(set(y_true) | set(y_pred))
    target_names = [label_encoder.inverse_transform([i])[0] for i in present]
    report = classification_report(y_true, y_pred, labels=present, target_names=target_names, digits=4, zero_division=0)

    # Distributions
    inv = label_encoder.inverse_transform
    true_labels = [inv([i])[0] for i in y_true]
    pred_labels = [inv([i])[0] for i in y_pred]
    true_dist = {}
    pred_dist = {}
    for lbl in true_labels:
        true_dist[lbl] = true_dist.get(lbl, 0) + 1
    for lbl in pred_labels:
        pred_dist[lbl] = pred_dist.get(lbl, 0) + 1

    # Confusion matrix (over present labels only)
    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_serializable = [list(map(int, row)) for row in cm]

    print(json.dumps({
        "accuracy": acc,
        "present_labels": target_names,
        "true_distribution": true_dist,
        "pred_distribution": pred_dist,
        "confusion_matrix": cm_serializable
    }, indent=2))
    print("\n" + report)


if __name__ == "__main__":
    csv = os.environ.get("EVAL_CSV", os.path.join("data", "processed", "all_cases.csv"))
    evaluate(csv)
