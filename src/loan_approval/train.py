from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier

from .data import load_csv
from .preprocess import (
    DataFrameSelector,
    FeatureAdder,
    Pipeline as ImbPipeline,
    build_preprocessor,
)


@dataclass
class TrainConfig:
    data_path: str
    target_col: str
    models: List[str]
    use_smote: bool
    artifact_path: str
    output_dir: str
    test_size: float
    val_size: float
    random_state: int


COMMON_POSITIVE_WORDS = {"y", "yes", "approved", "1", 1, True}


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype == "O":
        y_lower = y.astype(str).str.lower().str.strip()
        mapped = y_lower.apply(lambda v: 1 if v in COMMON_POSITIVE_WORDS else (0 if v in {"n", "no", "not approved", "0", 0, False} else np.nan))
        if mapped.isna().any():
            # Fallback: make the majority class negative, minority positive by alphabetical order
            vals = y_lower.value_counts(dropna=True)
            if len(vals) == 2:
                pos_label = vals.index[0]
                mapped = (y_lower == pos_label).astype(int)
            else:
                # Last resort: treat unique max label as positive, others 0
                mapped = (y_lower == vals.idxmax()).astype(int)
        return mapped.astype(int)
    # Numeric
    unique_vals = sorted(pd.Series(y.unique()).dropna().tolist())
    if set(unique_vals).issubset({0, 1}):
        return y.astype(int)
    # Binarize around median if continuous (not ideal, but a guard)
    median_val = np.median(y.astype(float))
    return (y.astype(float) >= median_val).astype(int)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float, random_state: int) -> Tuple:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_size_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_full_pipeline(
    X_sample: pd.DataFrame,
    use_smote: bool,
    model_name: str,
    random_state: int,
) -> Tuple[ImbPipeline, Dict[str, List]]:
    preprocessor, num_cols, cat_cols = build_preprocessor(X_sample)
    steps = [
        ("features", FeatureAdder()),
        ("selector", DataFrameSelector(columns=list(X_sample.columns))),
        ("preprocess", preprocessor),
    ]

    if use_smote:
        from imblearn.over_sampling import SMOTE
        steps.append(("smote", SMOTE(random_state=random_state)))

    if model_name == "log_reg":
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        steps.append(("clf", clf))
        param_grid = {
            "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "clf__penalty": ["l1", "l2"],
        }
    elif model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
        steps.append(("clf", clf))
        param_grid = {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [None, 6, 12, 18],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipe = ImbPipeline(steps)
    return pipe, param_grid


def evaluate_and_plot(model, X, y, split_name: str, out_dir: str) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob)) if y_prob is not None else float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")

    report = classification_report(y, y_pred, output_dict=True)
    metrics.update({
        f"precision_{split_name}": float(report["weighted avg"]["precision"]),
        f"recall_{split_name}": float(report["weighted avg"]["recall"]),
        f"f1_{split_name}": float(report["weighted avg"]["f1-score"]),
    })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"classification_report_{split_name}.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{split_name}.png"))
    plt.close()

    # ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y, y_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title(f"ROC Curve - {split_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_curve_{split_name}.png"))
        plt.close()

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y, y_prob)
        PrecisionRecallDisplay(precision=precision, recall=recall).plot()
        plt.title(f"Precision-Recall Curve - {split_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_curve_{split_name}.png"))
        plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a loan approval model")
    parser.add_argument("--data-path", required=True, help="CSV path or URL (Google Drive supported)")
    parser.add_argument("--target-col", default="Loan_Status", help="Target column name")
    parser.add_argument("--models", default="log_reg,rf", help="Comma-separated list: log_reg,rf")
    parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE oversampling")
    parser.add_argument("--artifact-path", default="/workspace/artifacts/loan_model.joblib", help="Path to save model artifact")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory to save reports and plots")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        target_col=args.target_col,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        use_smote=bool(args.use_smote),
        artifact_path=args.artifact_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    ensure_dirs(cfg.artifact_path)
    ensure_dir(cfg.output_dir)

    df = load_csv(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset. Columns: {list(df.columns)}")

    df = df.drop_duplicates()

    y_raw = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    # Basic EDA outputs
    with open(os.path.join(cfg.output_dir, "data_summary.txt"), "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write(str(df.dtypes))
        f.write("\n\nMissing values per column:\n")
        f.write(str(df.isnull().sum()))
        f.write("\n\nTarget distribution:\n")
        f.write(str(y_raw.value_counts(normalize=True)))

    y = normalize_target(y_raw)
    if y.isna().any():
        # Drop rows with un-mappable targets
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask].astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, cfg.test_size, cfg.val_size, cfg.random_state
    )

    candidates: Dict[str, Dict] = {}
    for model_name in cfg.models:
        pipe, param_grid = build_full_pipeline(
            X_sample=X_train, use_smote=cfg.use_smote, model_name=model_name, random_state=cfg.random_state
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        grid.fit(X_train, y_train)

        val_metrics = evaluate_and_plot(grid.best_estimator_, X_val, y_val, split_name=f"val_{model_name}", out_dir=cfg.output_dir)
        candidates[model_name] = {
            "search": grid,
            "val_roc_auc": val_metrics.get("roc_auc", float("nan")),
        }

    # Choose best by validation ROC-AUC
    best_name = max(candidates.keys(), key=lambda k: (candidates[k]["val_roc_auc"]))
    best_search: GridSearchCV = candidates[best_name]["search"]

    # Refit on train+val
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    best_estimator = best_search.best_estimator_
    best_params = best_search.best_params_

    final_model = clone(best_estimator)
    final_model.set_params(**best_params)
    final_model.fit(X_trainval, y_trainval)

    # Evaluate on test
    test_metrics = evaluate_and_plot(final_model, X_test, y_test, split_name="test", out_dir=cfg.output_dir)

    # Persist model
    joblib.dump(final_model, cfg.artifact_path)

    # Save metrics summary
    summary = {
        "best_model": best_name,
        "best_params": best_params,
        "val_roc_auc_by_model": {k: float(v["val_roc_auc"]) for k, v in candidates.items()},
        "test_metrics": test_metrics,
        "artifact_path": cfg.artifact_path,
    }
    with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save a processed CSV (engineered raw features only)
    from .features import add_engineered_features

    processed = add_engineered_features(X.copy())
    processed[cfg.target_col] = y.values
    processed.to_csv(os.path.join(cfg.output_dir, "processed_dataset.csv"), index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()