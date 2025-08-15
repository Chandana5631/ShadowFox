from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(data_path: str | Path, dtype: Optional[dict] = None) -> pd.DataFrame:
	path = Path(data_path)
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found at: {path}")
	return pd.read_csv(path, dtype=dtype)


def load_from_gdrive_file_id(file_id: str, dtype: Optional[dict] = None) -> pd.DataFrame:
	"""Load a CSV from Google Drive using the file's public `file_id`.

	The URL format uses the standard uc export endpoint. The file must be shared publicly
	or at least accessible with the link.
	"""
	url = f"https://drive.google.com/uc?export=download&id={file_id}"
	return pd.read_csv(url, dtype=dtype)