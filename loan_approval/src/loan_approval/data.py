import os
import re
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def _maybe_extract_gdrive_id(url_or_id: str) -> Optional[str]:
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
        return url_or_id
    patterns = [
        r"drive.google.com/file/d/([A-Za-z0-9_-]{20,})",
        r"drive.google.com/open\?id=([A-Za-z0-9_-]{20,})",
        r"drive.google.com/uc\?id=([A-Za-z0-9_-]{20,})",
        r"drive.google.com/uc\?export=download&id=([A-Za-z0-9_-]{20,})",
    ]
    for pat in patterns:
        match = re.search(pat, url_or_id)
        if match:
            return match.group(1)
    return None


def _download_from_gdrive(url_or_id: str, output_path: str) -> str:
    try:
        import gdown  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gdown is required to download from Google Drive. Please install it or provide a local CSV path."
        ) from exc

    file_id = _maybe_extract_gdrive_id(url_or_id) or url_or_id
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdown.download(id=file_id, output=output_path, quiet=False)
    return output_path


def load_dataset(data_source: str, cache_dir: str = "/tmp/loan_approval_cache") -> pd.DataFrame:
    if data_source.startswith("http") or "drive.google.com" in data_source or _maybe_extract_gdrive_id(data_source):
        os.makedirs(cache_dir, exist_ok=True)
        cached_path = os.path.join(cache_dir, "dataset.csv")
        csv_path = _download_from_gdrive(data_source, cached_path)
        return pd.read_csv(csv_path)
    if not os.path.exists(data_source):
        raise FileNotFoundError(f"Data source not found: {data_source}")
    return pd.read_csv(data_source)


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    temp_size = val_size + test_size
    df_train, df_temp = train_test_split(
        df, stratify=df[target_column], test_size=temp_size, random_state=random_state
    )
    relative_val_size = val_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_temp,
        stratify=df_temp[target_column],
        test_size=(1 - relative_val_size),
        random_state=random_state,
    )
    return df_train, df_val, df_test