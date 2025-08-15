from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessing_pipeline


def _safe_xgb_classifier():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        return None


def _safe_lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            n_jobs=-1,
        )
    except Exception:
        return None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features,
    categorical_features,
    model_type: str = "logistic_regression",
    use_class_weight: bool = True,
    perform_search: bool = True,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    candidate_model: Optional[object]
    param_grid = None

    if model_type == "logistic_regression":
        base = LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced" if use_class_weight else None)
        param_grid = {
            "model__C": [0.1, 1.0, 3.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
        }
        candidate_model = base
    elif model_type == "random_forest":
        base = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced" if use_class_weight else None,
            random_state=random_state,
        )
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        }
        candidate_model = base
    elif model_type == "xgboost":
        candidate_model = _safe_xgb_classifier()
        param_grid = None
        if candidate_model is None:
            raise ValueError("XGBoost is not available. Install xgboost or choose another model.")
    elif model_type == "lightgbm":
        candidate_model = _safe_lgbm_classifier()
        param_grid = None
        if candidate_model is None:
            raise ValueError("LightGBM is not available. Install lightgbm or choose another model.")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", candidate_model)])

    if perform_search and param_grid:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        best_pipe = search.best_estimator_
    else:
        pipe.fit(X, y)
        best_pipe = pipe

    y_prob = best_pipe.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y.values, y_prob, y_pred)
    return best_pipe, metrics