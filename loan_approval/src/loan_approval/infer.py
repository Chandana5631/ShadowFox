from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from .config import POSITIVE_LABEL


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference with a saved loan approval pipeline")
	parser.add_argument("--model", dest="model_path", type=str, required=True, help="Path to .joblib pipeline")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--json", dest="json_path", type=str, help="Path to JSON file with records or {records: [...]} ")
	group.add_argument("--stdin", action="store_true", help="Read JSON from STDIN")
	parser.add_argument("--as-labels", action="store_true", help="Map numeric predictions to {N,Y} labels")
	return parser.parse_args()


def load_payload(args: argparse.Namespace) -> Dict[str, Any] | List[Dict[str, Any]]:
	if args.json_path:
		path = Path(args.json_path)
		if not path.exists():
			raise FileNotFoundError(f"JSON input not found: {path}")
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	else:
		data = sys.stdin.read()
		return json.loads(data)


def normalize_records(payload: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	if isinstance(payload, dict) and "records" in payload:
		records = payload["records"]
	elif isinstance(payload, list):
		records = payload
	else:
		raise ValueError("Input JSON must be a list of records or an object with 'records'.")
	if not isinstance(records, list) or not records:
		raise ValueError("No records provided for prediction")
	return records


def map_to_labels(predictions: List[int]) -> List[str]:
	return [POSITIVE_LABEL if int(p) == 1 else ("N" if POSITIVE_LABEL == "Y" else f"not_{POSITIVE_LABEL}") for p in predictions]


def main() -> None:
	args = parse_args()
	pipeline = joblib.load(args.model_path)

	payload = load_payload(args)
	records = normalize_records(payload)
	df = pd.DataFrame.from_records(records)

	preds = pipeline.predict(df)
	try:
		probas = pipeline.predict_proba(df)
		if probas.shape[1] == 2:
			probas = probas[:, 1]
		else:
			probas = None
	except Exception:
		probas = None

	preds_list = [int(p) if not isinstance(p, (str, bool)) else p for p in preds.tolist()]
	if args.as_labels:
		preds_out = map_to_labels(preds_list)
	else:
		preds_out = preds_list

	result = {
		"predictions": preds_out,
		"probabilities": (probas.tolist() if hasattr(probas, "tolist") else None),
	}
	print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
	main()