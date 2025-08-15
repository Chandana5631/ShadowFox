from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

try:
	from xgboost import XGBClassifier  # type: ignore
	has_xgb = True
except Exception:
	has_xgb = False

try:
	from lightgbm import LGBMClassifier  # type: ignore
	has_lgbm = True
except Exception:
	has_lgbm = False


@dataclass
class SplitData:
	X_train: pd.DataFrame
	y_train: np.ndarray
	X_val: pd.DataFrame
	y_val: np.ndarray
	X_test: pd.DataFrame
	y_test: np.ndarray


def stratified_train_val_test_split(
	X: pd.DataFrame,
	y: np.ndarray,
	test_size: float,
	val_size: float,
	random_state: int = 42,
) -> SplitData:
	sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
	train_idx, test_idx = next(sss_outer.split(X, y))
	X_trainval, X_test = X.iloc[train_idx], X.iloc[test_idx]
	y_trainval, y_test = y[train_idx], y[test_idx]

	sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
	train_idx, val_idx = next(sss_inner.split(X_trainval, y_trainval))
	X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
	y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

	return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def build_logistic_regression() -> LogisticRegression:
	return LogisticRegression(max_iter=200, class_weight="balanced")


def build_random_forest() -> RandomForestClassifier:
	return RandomForestClassifier(
		n_estimators=400,
		max_depth=None,
		random_state=42,
		class_weight="balanced_subsample",
	)


def build_xgb_classifier() -> Optional[object]:
	if not has_xgb:
		return None
	return XGBClassifier(
		n_estimators=300,
		max_depth=5,
		learning_rate=0.08,
		subsample=0.9,
		colsample_bytree=0.9,
		eval_metric="logloss",
		random_state=42,
	)


def build_lgbm_classifier() -> Optional[object]:
	if not has_lgbm:
		return None
	return LGBMClassifier(
		n_estimators=500,
		learning_rate=0.05,
		num_leaves=31,
		subsample=0.9,
		colsample_bytree=0.9,
		random_state=42,
	)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
	metrics: Dict[str, float] = {}
	metrics["accuracy"] = accuracy_score(y_true, y_pred)
	metrics["precision"] = precision_score(y_true, y_pred, zero_division=0, pos_label=1)
	metrics["recall"] = recall_score(y_true, y_pred, zero_division=0, pos_label=1)
	metrics["f1"] = f1_score(y_true, y_pred, zero_division=0, pos_label=1)
	if y_proba is not None:
		try:
			metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
		except Exception:
			metrics["roc_auc"] = float("nan")
	return metrics