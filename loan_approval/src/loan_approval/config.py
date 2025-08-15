from __future__ import annotations

from typing import List

# Core dataset configuration
TARGET_COLUMN: str = "Loan_Status"
ID_COLUMNS: List[str] = ["Loan_ID"]
POSITIVE_LABEL: str = "Y"

# Splits
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.15  # yields 70/15/15 with VAL_SIZE ~ 0.17647 on the remaining 85%
VAL_SIZE: float = 0.17647

# Preprocessing defaults
DEFAULT_NUMERIC_IMPUTER_STRATEGY: str = "median"
DEFAULT_CATEGORICAL_IMPUTER_STRATEGY: str = "most_frequent"
SCALE_NUMERIC: bool = True

# Hints for common columns (optional; detection is automatic)
CATEGORICAL_COLUMNS_HINT: List[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_COLUMNS_HINT: List[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Engineered features (added if source columns exist)
ENGINEERED_COLUMNS: List[str] = [
    "TotalIncome",           # ApplicantIncome + CoapplicantIncome
    "DebtToIncomeRatio",     # LoanAmount / max(TotalIncome, 1)
    "LoanAmountLog",         # log1p(LoanAmount)
]