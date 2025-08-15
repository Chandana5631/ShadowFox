from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
	RANDOM_STATE,
	TARGET_COLUMN,
	TEST_SIZE,
	VAL_SIZE,
	POSITIVE_LABEL,
)
from .data_loader import load_csv, load_from_gdrive_file_id
from .modeling import (
	build_lgbm_classifier,
	build_logistic_regression,
	build_random_forest,
	build_xgb_classifier,
	compute_metrics,
	stratified_train_val_test_split,
)
from .preprocessing import make_dataset, make_pipeline


MODEL_MAP = {
	"lr": build_logistic_regression,
	"rf": build_random_forest,
	"xgb": build_xgb_classifier,
	"lgbm": build_lgbm_classifier,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train loan approval model")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--data", type=str, help="Absolute path to CSV dataset")
	group.add_argument("--gdrive-file-id", type=str, help="Google Drive file id of CSV")

	parser.add_argument("--target", type=str, default=TARGET_COLUMN, help="Target column name")
	parser.add_argument("--id-col", type=str, default="Loan_ID", help="ID column to drop if present")
	parser.add_argument("--model", type=str, default="rf", choices=list(MODEL_MAP.keys()))
	parser.add_argument("--use-smote", action="store_true", help="Enable SMOTE oversampling")
	parser.add_argument("--test-size", type=float, default=TEST_SIZE)
	parser.add_argument("--val-size", type=float, default=VAL_SIZE)
	parser.add_argument("--save-path", type=str, default=str(Path("models/loan_approval_pipeline.joblib").absolute()))
	parser.add_argument("--metrics-path", type=str, default=str(Path("models/metrics.json").absolute()))
	return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
	if args.data:
		return load_csv(args.data)
	return load_from_gdrive_file_id(args.gdrive_file_id)


def prepare_labels(y_raw: np.ndarray, positive_label: str = POSITIVE_LABEL) -> np.ndarray:
	# Map labels to binary {0,1}
	if y_raw is None:
		raise ValueError("Target column not found in dataset")
	labels = pd.Series(y_raw)
	unique_values = sorted(labels.dropna().unique())
	if len(unique_values) <= 2 and positive_label in unique_values:
		return (labels == positive_label).astype(int).values
	# Fallback: if labels already numeric-ish
	try:
		return labels.astype(int).values
	except Exception:
		raise ValueError(
			f"Unexpected target values {unique_values}. Provide binary target with positive label '{positive_label}'."
		)


def main() -> None:
	args = parse_args()

	df = load_dataset(args)
	if args.id_col in df.columns:
		df = df.drop(columns=[args.id_col])

	X, y_raw, numeric_features, categorical_features = make_dataset(df, target=args.target)
	y = prepare_labels(y_raw)

	splits = stratified_train_val_test_split(
		X, y, test_size=args.test_size, val_size=args.val_size, random_state=RANDOM_STATE
	)

	model_builder = MODEL_MAP[args.model]
	estimator = model_builder()
	if estimator is None:
		raise RuntimeError(f"Selected model '{args.model}' requires an optional dependency.")

	pipeline = make_pipeline(
		estimator=estimator,
		use_smote=args.use_smote,
		numeric_features=numeric_features,
		categorical_features=categorical_features,
	)

	pipeline.fit(splits.X_train, splits.y_train)

	def predict_proba_safe(X_eval):
		try:
			probas = pipeline.predict_proba(X_eval)
			if hasattr(probas, "shape") and probas.shape[1] == 2:
				return probas[:, 1]
			return None
		except Exception:
			return None

	y_pred_val = pipeline.predict(splits.X_val)
	y_proba_val = predict_proba_safe(splits.X_val)
	metrics_val = compute_metrics(splits.y_val, y_pred_val, y_proba_val)

	y_pred_test = pipeline.predict(splits.X_test)
	y_proba_test = predict_proba_safe(splits.X_test)
	metrics_test = compute_metrics(splits.y_test, y_pred_test, y_proba_test)

	Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(pipeline, args.save_path)

	all_metrics = {"val": metrics_val, "test": metrics_test}
	with open(args.metrics_path, "w", encoding="utf-8") as f:
		json.dump(all_metrics, f, indent=2)

	print("Saved model to:", args.save_path)
	print("Saved metrics to:", args.metrics_path)
	print("Validation metrics:", metrics_val)
	print("Test metrics:", metrics_test)


if __name__ == "__main__":
	main()