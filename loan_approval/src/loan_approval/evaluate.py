import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_and_save_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fig_dir: str,
    prefix: str,
    y_pred: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    try:
        roc_disp = RocCurveDisplay.from_predictions(y_true, y_prob)
        roc_disp.ax_.set_title("ROC Curve")
        plt.savefig(os.path.join(fig_dir, f"{prefix}_roc_curve.png"), bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    try:
        pr_disp = PrecisionRecallDisplay.from_predictions(y_true, y_prob)
        pr_disp.ax_.set_title("Precision-Recall Curve")
        plt.savefig(os.path.join(fig_dir, f"{prefix}_pr_curve.png"), bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    if y_pred is not None:
        try:
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(fig_dir, f"{prefix}_confusion_matrix.png"), bbox_inches="tight")
            plt.close()
        except Exception:
            pass