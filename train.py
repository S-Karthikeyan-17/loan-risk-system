#!/usr/bin/env python3
"""
Loan Default Risk — End-to-End Training Script

Implements:
- Data loading from provided Excel
- Preprocessing & feature engineering
- Model training (LogReg, RandomForest, XGBoost if available)
- Cross-validation & holdout evaluation
- Model selection on ROC-AUC
- Artifact persistence (model, metrics, reports, feature names)
- Visualizations (EDA + performance + feature importance)

Run:
  python train.py --data-path data/6S_AI_TASK-Loan_default_Loan_default.xlsx --out-dir artifacts

Author: (you)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Try to import XGBoost; if not installed, we’ll skip it gracefully.
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ----------------------------
# Config & Logging
# ----------------------------

@dataclass
class TrainConfig:
    data_path: Path
    out_dir: Path
    target_col: str = "Default"
    id_cols: Tuple[str, ...] = ("LoanID",)
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    n_jobs: int = -1
    max_plots: int = 30  # limit numeric distribution plots to avoid clutter


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ----------------------------
# Feature Engineering
# ----------------------------

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Adds engineered features using only raw numeric columns required.
    Works on pandas DataFrame BEFORE column transformer (so refer to original names).
    """

    def __init__(self):
        self.created_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.created_.clear()

        def safe_ratio(numer, denom, name):
            if denom in X.columns and numer in X.columns:
                X[name] = X[numer] / (X[denom].replace(0, np.nan) + 1e-6)
                X[name].replace([np.inf, -np.inf], np.nan, inplace=True)
                X[name].fillna(0.0, inplace=True)
                self.created_.append(name)

        # Ratios
        safe_ratio("Income", "LoanAmount", "Income_to_Loan_Ratio")
        safe_ratio("CreditScore", "LoanAmount", "Credit_per_LoanAmount")
        safe_ratio("LoanAmount", "NumCreditLines", "LoanAmount_per_CreditLine")

        # Tenure & interest derived features
        if "MonthsEmployed" in X.columns:
            X["EmploymentYears"] = X["MonthsEmployed"] / 12.0
            self.created_.append("EmploymentYears")

        if "InterestRate" in X.columns:
            X["HighInterestFlag"] = (X["InterestRate"] >= 15).astype(int)
            self.created_.append("HighInterestFlag")

        if "LoanTerm" in X.columns:
            X["TermYears"] = X["LoanTerm"] / 12.0
            self.created_.append("TermYears")

        # Age & DTIRatio buckets (coarse risk bands)
        if "Age" in X.columns:
            X["AgeBucket"] = pd.cut(
                X["Age"],
                bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
                labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )
            self.created_.append("AgeBucket")

        if "DTIRatio" in X.columns:
            X["DTI_Bucket"] = pd.cut(
                X["DTIRatio"],
                bins=[-np.inf, 0.2, 0.35, 0.5, 0.65, 0.8, np.inf],
                labels=["<=0.2", "0.21-0.35", "0.36-0.5", "0.51-0.65", "0.66-0.8", "0.8+"],
            )
            self.created_.append("DTI_Bucket")

        return X


# ----------------------------
# Utility: plotting helpers
# ----------------------------

def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()


def plot_class_balance(y: pd.Series, out: Path) -> None:
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["No Default (0)", "Default (1)"], counts.values)
    plt.title("Class Imbalance")
    plt.ylabel("Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    save_fig(out / "eda_class_balance.png")


def plot_numeric_dists(df: pd.DataFrame, numeric_cols: List[str], out: Path, max_plots: int) -> None:
    shown = 0
    for col in numeric_cols:
        if shown >= max_plots:
            break
        plt.figure(figsize=(5, 4))
        plt.hist(df[col].dropna().values, bins=30)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_fig(out / f"eda_dist_{col}.png")
        shown += 1


def plot_corr_heatmap(df: pd.DataFrame, numeric_cols: List[str], out: Path) -> None:
    if len(numeric_cols) == 0:
        return
    corr = df[numeric_cols].corr().fillna(0.0)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Numeric Feature Correlations")
    save_fig(out / "eda_correlation_heatmap.png")


def plot_roc_pr_curves(y_true, prob_dict: Dict[str, np.ndarray], out: Path) -> None:
    # ROC
    plt.figure(figsize=(6, 5))
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()
    save_fig(out / "perf_roc_curves.png")

    # PR
    plt.figure(figsize=(6, 5))
    for name, probs in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    save_fig(out / "perf_pr_curves.png")


def plot_risk_segmentation(y_true: np.ndarray, y_prob: np.ndarray, out: Path) -> None:
    """Decile bins of predicted risk vs. observed default rate."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["decile"] = pd.qcut(df["p"], q=10, labels=False, duplicates="drop")
    seg = df.groupby("decile").agg(count=("y", "size"), default_rate=("y", "mean")).reset_index()
    plt.figure(figsize=(7, 4))
    plt.bar(seg["decile"].astype(str), seg["default_rate"])
    plt.xlabel("Predicted Risk Decile (0=lowest risk)")
    plt.ylabel("Observed Default Rate")
    plt.title("Risk Segmentation by Predicted Probability")
    save_fig(out / "perf_risk_segmentation.png")


def plot_feature_importance(model: BaseEstimator, X: pd.DataFrame, out: Path, title: str) -> None:
    """Permutation importance on the holdout set for model-agnostic ranking."""
    try:
        r = permutation_importance(model, X, model.predict, n_repeats=10, random_state=42, n_jobs=-1, scoring="f1")
        importances = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)[:30]
        plt.figure(figsize=(8, 10))
        plt.barh(importances.index[::-1], importances.values[::-1])
        plt.title(f"Permutation Importance (top 30)\n{title}")
        plt.xlabel("Mean Importance (F1 scoring)")
        save_fig(out / "feat_importance_permutation.png")
    except Exception as e:
        logging.warning("Permutation importance failed: %s", e)


# ----------------------------
# Core Training
# ----------------------------

def load_data(cfg: TrainConfig) -> pd.DataFrame:
    logging.info("Loading data from %s", cfg.data_path)
    df = pd.read_excel(cfg.data_path)
    # Basic sanity
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found. Columns: {list(df.columns)}")
    return df


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_models(cfg: TrainConfig) -> Dict[str, BaseEstimator]:
    models: Dict[str, BaseEstimator] = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=cfg.n_jobs),
        "RF": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
        ),
    }
    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            eval_metric="logloss",
        )
    else:
        logging.warning("xgboost not installed; skipping XGB.")
    return models


def main(cfg: TrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load
    df = load_data(cfg)

    # Drop IDs
    for col in cfg.id_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Separate target early to avoid leakage
    y = df[cfg.target_col].astype(int).values
    X = df.drop(columns=[cfg.target_col])

    # Basic missing handling before FE (so bucketings work)
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(exclude=[np.number]).columns:
        X[col] = X[col].fillna(X[col].mode().iloc[0])

    # Feature engineering (adds columns, including new categoricals)
    fe = FeatureBuilder()
    X = fe.fit_transform(X)

    # Identify numeric & categorical cols (post-FE)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    logging.info("Numeric columns: %s", numeric_cols)
    logging.info("Categorical columns: %s", categorical_cols)

    # EDA snapshots
    plot_class_balance(pd.Series(y), plots_dir)
    plot_numeric_dists(X, numeric_cols, plots_dir, cfg.max_plots)
    plot_corr_heatmap(pd.concat([X[numeric_cols], pd.Series(y, name="Default")], axis=1), numeric_cols, plots_dir)

    # Train/test split (holdout)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # Preprocessor
    preprocessor = build_pipeline(numeric_cols, categorical_cols)

    # Models
    models = get_models(cfg)

    # CV and holdout evaluation
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    cv_summary: Dict[str, Dict[str, float]] = {}
    holdout_probs: Dict[str, np.ndarray] = {}
    holdout_preds: Dict[str, np.ndarray] = {}

    best_name = None
    best_auc = -1.0
    best_pipe: Optional[Pipeline] = None

    for name, est in models.items():
        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", est),
        ])

        # CV on training set
        cv_auc = cross_val_score(pipe, X_train_df, y_train, cv=skf, scoring="roc_auc", n_jobs=cfg.n_jobs).mean()
        cv_f1 = cross_val_score(pipe, X_train_df, y_train, cv=skf, scoring="f1", n_jobs=cfg.n_jobs).mean()
        cv_acc = cross_val_score(pipe, X_train_df, y_train, cv=skf, scoring="accuracy", n_jobs=cfg.n_jobs).mean()

        cv_summary[name] = {"cv_auc": cv_auc, "cv_f1": cv_f1, "cv_acc": cv_acc}
        logging.info("CV [%s] AUC=%.4f | F1=%.4f | ACC=%.4f", name, cv_auc, cv_f1, cv_acc)

        # Fit and evaluate on holdout
        pipe.fit(X_train_df, y_train)
        prob = pipe.predict_proba(X_test_df)[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc_h = roc_auc_score(y_test, prob)
        f1_h = f1_score(y_test, pred)
        acc_h = accuracy_score(y_test, pred)
        logging.info("Holdout [%s] AUC=%.4f | F1=%.4f | ACC=%.4f", name, auc_h, f1_h, acc_h)

        holdout_probs[name] = prob
        holdout_preds[name] = pred

        if auc_h > best_auc:
            best_auc = auc_h
            best_name = name
            best_pipe = pipe

    assert best_pipe is not None and best_name is not None

    # Save best model pipeline (includes preprocessor)
    model_path = cfg.out_dir / "model.pkl"
    joblib.dump(best_pipe, model_path)
    logging.info("Best model: %s (AUC=%.4f) saved to %s", best_name, best_auc, model_path)

    # Build feature names after preprocessor for inspection
    # We can get names by fitting a clone of preprocessor:
    preprocessor.fit(X_train_df)
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(x) for x in feature_names]
    except Exception:
        # Fallback if version mismatch
        feature_names = [f"f_{i}" for i in range(len(preprocessor.transform(X_train_df)[0]))]

    with open(cfg.out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Persist metrics & reports
    metrics = {
        "best_model": best_name,
        "cv": cv_summary,
        "holdout": {
            m: {
                "auc": roc_auc_score(y_test, holdout_probs[m]),
                "f1": f1_score(y_test, holdout_preds[m]),
                "acc": accuracy_score(y_test, holdout_preds[m]),
            } for m in holdout_probs
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_balance": {
            "train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
            "test": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        }
    }
    with open(cfg.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save classification report for best model
    best_pred = holdout_preds[best_name]
    report_txt = classification_report(y_test, best_pred, digits=4)
    with open(cfg.out_dir / "classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_name}\n\n")
        f.write(report_txt)

    # Visualizations: ROC/PR for all models
    plot_roc_pr_curves(y_test, holdout_probs, plots_dir)

    # Risk segmentation for best model
    plot_risk_segmentation(y_test, holdout_probs[best_name], plots_dir)

    # Feature importance (permutation) on holdout data using the *trained* pipeline.
    # We need transformed X to get correct feature space. We'll rebuild a model-only importance
    # by wrapping predict to work on transformed X.
    # Extract the fitted preprocessor & classifier:
    fitted_pre: ColumnTransformer = best_pipe.named_steps["pre"]
    clf = best_pipe.named_steps["clf"]
    X_test_trans = fitted_pre.transform(X_test_df)
    X_test_df_trans = pd.DataFrame(X_test_trans, columns=feature_names)

    # Build a small wrapper that exposes predict for permutation_importance
    class ModelWrapper(BaseEstimator):
        def __init__(self, estimator):
            self.estimator = estimator
        def fit(self, X, y=None): return self
        def predict(self, X): return self.estimator.predict(X)

    try:
        plot_feature_importance(ModelWrapper(clf), X_test_df_trans, plots_dir, f"Best: {best_name}")
    except Exception as e:
        logging.warning("Could not plot feature importance: %s", e)

    logging.info("Training complete. Artifacts saved to: %s", cfg.out_dir)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Loan Default Risk Model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/6S_AI_TASK-Loan_default_Loan_default.xlsx",
        help="Path to Excel dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Directory to save models/plots/reports",
    )
    args = parser.parse_args()
    return TrainConfig(
        data_path=Path(args.data_path),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    setup_logging()
    cfg = parse_args()
    main(cfg)
