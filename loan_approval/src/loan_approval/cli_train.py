import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .data import load_dataset, stratified_split
from .preprocessing import infer_feature_types, engineer_optional_features
from .model import train_model
from .evaluate import save_metrics, plot_and_save_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Loan Approval model")
    parser.add_argument("--data-source", required=True, help="CSV path or Google Drive URL/file-id")
    parser.add_argument("--target-column", default="Loan_Status", help="Target column name")
    parser.add_argument("--model-type", default="logistic_regression", choices=[
        "logistic_regression", "random_forest", "xgboost", "lightgbm"
    ])
    parser.add_argument("--no-search", action="store_true", help="Disable hyperparameter search")
    parser.add_argument("--output-model", default="outputs/models/loan_model.pkl", help="Path to save model .pkl")
    parser.add_argument("--metrics-json", default="outputs/reports/metrics.json", help="Path to save metrics JSON")
    parser.add_argument("--fig-dir", default="outputs/figures", help="Directory to save evaluation figures")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data_source)

    if args.target_column not in df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in dataset. Columns: {list(df.columns)}")

    # Ensure binary numeric target (0/1)
    if not pd.api.types.is_numeric_dtype(df[args.target_column]):
        unique_vals = list(pd.Series(df[args.target_column]).dropna().unique())
        if len(unique_vals) == 2:
            positive = unique_vals[0]
            mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
            df[args.target_column] = df[args.target_column].map(mapping)
        else:
            raise ValueError(
                f"Target column must be binary. Found values: {unique_vals}. Please pre-process or specify a valid target."
            )

    df = engineer_optional_features(df)

    numeric_features, categorical_features = infer_feature_types(df, args.target_column)

    df_train, df_val, df_test = stratified_split(
        df, target_column=args.target_column,
        train_size=args.train_size, val_size=args.val_size, test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = df_train.drop(columns=[args.target_column])
    y_train = df_train[args.target_column]
    X_val = df_val.drop(columns=[args.target_column])
    y_val = df_val[args.target_column]
    X_test = df_test.drop(columns=[args.target_column])
    y_test = df_test[args.target_column]

    pipeline, train_metrics = train_model(
        pd.concat([X_train, X_val], axis=0),
        pd.concat([y_train, y_val], axis=0),
        numeric_features,
        categorical_features,
        model_type=args.model_type,
        perform_search=not args.no_search,
        random_state=args.random_state,
    )

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    from .model import compute_metrics

    test_metrics = compute_metrics(y_test.values, y_prob, y_pred)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(pipeline, args.output_model)

    save_metrics(test_metrics, args.metrics_json)
    plot_and_save_curves(y_test.values, y_prob, args.fig_dir, prefix="test", y_pred=y_pred)

    print(json.dumps({"train_metrics": train_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()