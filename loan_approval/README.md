# Loan Approval Prediction

A complete ML pipeline to predict loan approval decisions from applicant and loan attributes. Includes EDA, preprocessing + modeling pipeline, evaluation, model persistence, and an optional FastAPI service.

## Quickstart

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train a model from a CSV path or Google Drive URL/file-id:

```bash
python -m src.loan_approval.cli_train \
  --data-source /path/to/loan_data.csv \
  --target-column Loan_Status \
  --model-type logistic_regression \
  --output-model /workspace/loan_approval/outputs/models/loan_model.pkl
```

3. Serve the trained model via FastAPI:

```bash
uvicorn src.loan_approval.api.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
loan_approval/
  data/
  notebooks/
  outputs/
    models/
    processed/
    reports/
    figures/
  src/
    loan_approval/
      api/
        app.py
      __init__.py
      data.py
      preprocessing.py
      model.py
      evaluate.py
      cli_train.py
requirements.txt
README.md
```

## Notes
- The training CLI accepts either a local CSV path or a Google Drive sharing URL/file-id (via gdown).
- The saved `.pkl` artifact contains the full preprocessing + model pipeline and can consume raw feature dicts.
- See the `notebooks/` folder for an EDA + modeling walkthrough.