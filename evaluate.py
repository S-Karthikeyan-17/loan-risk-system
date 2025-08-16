#!/usr/bin/env python3
"""
Loan Default Risk — Evaluation Script

Usage:
  python evaluate.py --data-path data/6S_AI_TASK-Loan_default_Loan_default.xlsx --model-path artifacts/model.pkl
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------
# Feature Engineering (same as train.py)
# -------------------------
class FeatureBuilder:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def safe_ratio(numer, denom, name):
            if numer in X.columns and denom in X.columns:
                X[name] = X[numer] / (X[denom].replace(0, np.nan) + 1e-6)
                X[name].replace([np.inf, -np.inf], np.nan, inplace=True)
                X[name].fillna(0.0, inplace=True)

        safe_ratio("Income", "LoanAmount", "Income_to_Loan_Ratio")
        safe_ratio("CreditScore", "LoanAmount", "Credit_per_LoanAmount")
        safe_ratio("LoanAmount", "NumCreditLines", "LoanAmount_per_CreditLine")

        if "MonthsEmployed" in X.columns:
            X["EmploymentYears"] = X["MonthsEmployed"] / 12.0

        if "InterestRate" in X.columns:
            X["HighInterestFlag"] = (X["InterestRate"] >= 15).astype(int)

        if "LoanTerm" in X.columns:
            X["TermYears"] = X["LoanTerm"] / 12.0

        if "Age" in X.columns:
            X["AgeBucket"] = pd.cut(
                X["Age"],
                bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
                labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )

        if "DTIRatio" in X.columns:
            X["DTI_Bucket"] = pd.cut(
                X["DTIRatio"],
                bins=[-np.inf, 0.2, 0.35, 0.5, 0.65, 0.8, np.inf],
                labels=["<=0.2", "0.21-0.35", "0.36-0.5", "0.51-0.65", "0.66-0.8", "0.8+"],
            )

        # Fill NA
        for c in X.select_dtypes(include=[np.number]).columns:
            X[c] = X[c].fillna(X[c].median())
        for c in X.select_dtypes(exclude=[np.number]).columns:
            X[c] = X[c].fillna(X[c].mode().iloc[0])

        return X

# -------------------------
# CLI args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Loan Default Model")
    parser.add_argument("--data-path", type=str, default="data/6S_AI_TASK-Loan_default_Loan_default.xlsx")
    parser.add_argument("--model-path", type=str, default="artifacts/model.pkl")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    return parser.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("📥 Loading dataset...")
    df = pd.read_excel(data_path)

    if "Default" not in df.columns:
        raise ValueError("Dataset must contain target column 'Default'")

    y = df["Default"].astype(int).values
    X = df.drop(columns=["Default"])

    # Apply feature engineering
    logging.info("⚙️ Applying feature engineering...")
    X = FeatureBuilder().transform(X)

    # Load trained model
    logging.info(f"📥 Loading model from {model_path}")
    model = joblib.load(model_path)

    # Predict
    logging.info("🧮 Evaluating model...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
    }

    logging.info(f"✅ Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"✅ F1 Score: {metrics['f1']:.4f}")
    logging.info(f"✅ ROC-AUC: {metrics['roc_auc']:.4f}")

    # Save metrics JSON
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report
    report = classification_report(y, y_pred, digits=4)
    with open(out_dir / "eval_classification_report.txt", "w") as f:
        f.write(report)

    logging.info("📊 Evaluation results saved in artifacts/")

if __name__ == "__main__":
    main()
