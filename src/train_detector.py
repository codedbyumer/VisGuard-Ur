"""Train the lightweight UrduGuard-compatible detector.

A character n-gram model is deliberately retained because it handles Urdu script
and Roman Urdu spelling variation without a downloaded transformer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "visguard_ur_dataset.csv"
MODEL_PATH = ROOT / "results" / "visguard_detector.joblib"
METRICS_PATH = ROOT / "results" / "detector_metrics.json"


def make_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        strip_accents=None,
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=2500, C=5.0, random_state=42)
    return vectorizer, clf


def train_detector() -> dict:
    df = pd.read_csv(DATASET_PATH).drop_duplicates("text").reset_index(drop=True)
    df["y"] = (df["label"] == "adversarial").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.30, random_state=42, stratify=df["y"]
    )
    vectorizer, clf = make_model()
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_unique_prompts": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification_report": classification_report(y_test, y_pred, target_names=["benign", "adversarial"], output_dict=True),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clf": clf, "threshold": 0.5}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, y_pred, target_names=["benign", "adversarial"], zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    return metrics


def predict_texts(texts: list[str], threshold: float = 0.5) -> list[dict]:
    bundle = joblib.load(MODEL_PATH)
    vectorizer, clf = bundle["vectorizer"], bundle["clf"]
    probabilities = clf.predict_proba(vectorizer.transform([t or "" for t in texts]))[:, 1]
    return [{"risk_score": round(float(p), 5), "blocked": bool(p >= threshold)} for p in probabilities]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    train_detector()
    if args.threshold != 0.5:
        bundle = joblib.load(MODEL_PATH)
        bundle["threshold"] = args.threshold
        joblib.dump(bundle, MODEL_PATH)


if __name__ == "__main__":
    main()
