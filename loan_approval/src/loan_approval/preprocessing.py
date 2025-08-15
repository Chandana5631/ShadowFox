from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
	CATEGORICAL_COLUMNS_HINT,
	NUMERIC_COLUMNS_HINT,
	DEFAULT_CATEGORICAL_IMPUTER_STRATEGY,
	DEFAULT_NUMERIC_IMPUTER_STRATEGY,
	SCALE_NUMERIC,
)
from .features import apply_all_engineering, FeatureEngineeringTransformer


def detect_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
	candidate_categorical: List[str] = []
	candidate_numeric: List[str] = []
	for col in df.columns:
		if col == target:
			continue
		if df[col].dtype == "object" or df[col].dtype.name.startswith("category"):
			candidate_categorical.append(col)
		else:
			candidate_numeric.append(col)

	for col in CATEGORICAL_COLUMNS_HINT:
		if col in df.columns and col not in candidate_categorical and col != target:
			candidate_categorical.append(col)
	for col in NUMERIC_COLUMNS_HINT:
		if col in df.columns and col not in candidate_numeric and col != target:
			candidate_numeric.append(col)

	candidate_categorical = list(dict.fromkeys(candidate_categorical))
	candidate_numeric = list(dict.fromkeys(candidate_numeric))
	return candidate_numeric, candidate_categorical


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
	# We apply feature engineering first, then column-wise transforms
	feature_engineering = FeatureEngineeringTransformer()

	numeric_steps: List[tuple] = [
		("imputer", SimpleImputer(strategy=DEFAULT_NUMERIC_IMPUTER_STRATEGY)),
	]
	if SCALE_NUMERIC and numeric_features:
		numeric_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy=DEFAULT_CATEGORICAL_IMPUTER_STRATEGY)),
		("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	])

	column_transformer = ColumnTransformer(
		transformers=[
			("num", Pipeline(steps=numeric_steps), numeric_features + ["TotalIncome", "LoanAmountLog", "DebtToIncomeRatio"]),
			("cat", categorical_transformer, categorical_features),
		],
		remainder="drop",
	)

	full_preprocess = Pipeline(steps=[
		("features", feature_engineering),
		("columns", column_transformer),
	])
	return full_preprocess


def make_dataset(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
	df_engineered = apply_all_engineering(df)
	numeric_features, categorical_features = detect_column_types(df_engineered, target=target)

	X = df.drop(columns=[target]) if target in df.columns else df.copy()
	y = df[target].values if target in df.columns else None
	return X, y, numeric_features, categorical_features


def make_pipeline(estimator, use_smote: bool, numeric_features: List[str], categorical_features: List[str]):
	preprocessor = build_preprocessor(numeric_features, categorical_features)
	if use_smote:
		from imblearn.over_sampling import SMOTE  # type: ignore
		from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
		pipeline = ImbPipeline(steps=[
			("preprocess", preprocessor),
			("smote", SMOTE()),
			("model", estimator),
		])
	else:
		pipeline = Pipeline(steps=[
			("preprocess", preprocessor),
			("model", estimator),
		])
	return pipeline