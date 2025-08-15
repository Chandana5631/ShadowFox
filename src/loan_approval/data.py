from __future__ import annotations

import os
import tempfile
from typing import Optional

import pandas as pd


def _download_from_google_drive(url: str, output_path: str) -> Optional[str]:
    try:
        import gdown  # type: ignore
    except Exception:
        return None

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
        return output_path if os.path.exists(output_path) else None
    except Exception:
        return None


def load_csv(source: str) -> pd.DataFrame:
    """Load a CSV from a local path or URL. If it looks like Google Drive, try gdown.

    Args:
        source: Path or URL to CSV.
    Returns:
        DataFrame loaded from the source.
    """
    source = source.strip()

    if source.startswith("http") and ("drive.google.com" in source or "id=" in source):
        tmp_dir = os.path.join("/workspace", "data")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file = os.path.join(tmp_dir, "loan_data.csv")
        downloaded = _download_from_google_drive(source, tmp_file)
        if downloaded is not None:
            return pd.read_csv(downloaded)

    if source.startswith("http"):
        return pd.read_csv(source)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Data file not found: {source}")

    return pd.read_csv(source)