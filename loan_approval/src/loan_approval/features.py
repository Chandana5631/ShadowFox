from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def add_total_income(df: pd.DataFrame) -> pd.DataFrame:
	if {"ApplicantIncome", "CoapplicantIncome"}.issubset(df.columns):
		df = df.copy()
		df["TotalIncome"] = (
			df["ApplicantIncome"].fillna(0).astype(float)
			+ df["CoapplicantIncome"].fillna(0).astype(float)
		)
	return df


def add_loan_amount_log(df: pd.DataFrame) -> pd.DataFrame:
	if "LoanAmount" in df.columns:
		df = df.copy()
		df["LoanAmountLog"] = np.log1p(df["LoanAmount"].astype(float).clip(lower=0))
	return df


def add_debt_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
	if {"LoanAmount", "TotalIncome"}.issubset(df.columns):
		df = df.copy()
		total_income = df["TotalIncome"].astype(float).replace(0, np.nan)
		df["DebtToIncomeRatio"] = (
			df["LoanAmount"].astype(float) / total_income
		).fillna(0.0)
	return df


def apply_all_engineering(df: pd.DataFrame) -> pd.DataFrame:
	engineered = add_total_income(df)
	engineered = add_loan_amount_log(engineered)
	engineered = add_debt_to_income_ratio(engineered)
	return engineered


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self

	def transform(self, X):
		if isinstance(X, pd.DataFrame):
			return apply_all_engineering(X)
		# Convert to DataFrame conservatively if not already
		return apply_all_engineering(pd.DataFrame(X))