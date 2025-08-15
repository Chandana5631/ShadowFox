from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    candidate_categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    candidate_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in candidate_categorical if c != target_column]
    numeric_features = [c for c in candidate_numeric if c != target_column]
    return numeric_features, categorical_features


def engineer_optional_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    if {"ApplicantIncome", "CoapplicantIncome"}.issubset(engineered.columns):
        engineered["TotalIncome"] = (
            engineered["ApplicantIncome"].fillna(0) + engineered["CoapplicantIncome"].fillna(0)
        )

    if {"LoanAmount", "ApplicantIncome"}.issubset(engineered.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            engineered["DebtToIncome"] = np.where(
                engineered["ApplicantIncome"].fillna(0) > 0,
                engineered["LoanAmount"].fillna(0) / engineered["ApplicantIncome"].replace(0, np.nan),
                np.nan,
            )

    return engineered


def build_preprocessing_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor