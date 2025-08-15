from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline

from .features import add_engineered_features


class FeatureAdder:
    """Sklearn-compatible transformer to add engineered features consistently in train and inference."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(pd.DataFrame(X).copy())


class DataFrameSelector:
    """Selects the columns by name, robust to missing columns by filling with NaN.

    Useful when training and inference schemas differ slightly.
    """

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in self.columns:
            if col not in X_df.columns:
                X_df[col] = np.nan
        return X_df[self.columns]


def build_preprocessor(
    df_sample: pd.DataFrame,
    numeric_features_hint: Optional[List[str]] = None,
    categorical_features_hint: Optional[List[str]] = None,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Infer feature types and build a ColumnTransformer.

    If hints are provided, use them; otherwise infer numeric vs categorical by dtype.
    Returns the preprocessor and the final lists used.
    """
    df_enriched = add_engineered_features(df_sample)

    if numeric_features_hint is None or categorical_features_hint is None:
        inferred_categoricals = [
            c for c in df_enriched.columns if df_enriched[c].dtype == "object"
        ]
        inferred_numerics = [
            c for c in df_enriched.columns if c not in inferred_categoricals
        ]
    else:
        inferred_numerics = list(numeric_features_hint)
        inferred_categoricals = list(categorical_features_hint)

    numeric_transformer = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, inferred_numerics),
            ("cat", categorical_transformer, inferred_categoricals),
        ]
    )

    return preprocessor, inferred_numerics, inferred_categoricals


class Pipeline(ImbPipeline):
    """Thin alias to keep import familiarity from imblearn.pipeline."""

    pass