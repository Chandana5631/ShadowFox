from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain features to the input DataFrame, returning a new DataFrame.

    - TotalIncome = ApplicantIncome + CoapplicantIncome (if available)
    - DebtToIncomeRatio = LoanAmount / (TotalIncome + 1)
    """
    df_copy = df.copy()

    applicant_income = df_copy.get("ApplicantIncome")
    coapplicant_income = df_copy.get("CoapplicantIncome")

    if applicant_income is not None and coapplicant_income is not None:
        total_income = applicant_income.fillna(0) + coapplicant_income.fillna(0)
        df_copy["TotalIncome"] = total_income
    elif applicant_income is not None:
        df_copy["TotalIncome"] = applicant_income.fillna(0)
    else:
        # If ApplicantIncome is missing entirely, create TotalIncome with zeros to keep schema stable
        df_copy["TotalIncome"] = 0.0

    loan_amount = df_copy.get("LoanAmount")
    if loan_amount is not None:
        denom = df_copy["TotalIncome"].replace(0, np.nan)
        dti = loan_amount.astype(float) / (denom + 1)
        dti = dti.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df_copy["DebtToIncomeRatio"] = dti
    else:
        df_copy["DebtToIncomeRatio"] = 0.0

    return df_copy