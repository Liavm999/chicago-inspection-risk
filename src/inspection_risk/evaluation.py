"""Plots, segment-level error analysis, and permutation interpretation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

from inspection_risk.features import MODEL_FEATURES, TARGET


def plot_model_comparison(
    validation: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    labels = {
        "logistic_regression": "Logistic regression",
        "hist_gradient_boosting": "Gradient boosting",
    }
    for name, scores in probabilities.items():
        if name == "prevalence_baseline":
            continue
        precision, recall, _ = precision_recall_curve(validation[TARGET], scores)
        ax.plot(recall, precision, linewidth=2, label=labels[name])
    ax.axhline(
        validation[TARGET].mean(), color="grey", linestyle="--", label="Prevalence baseline"
    )
    ax.set(
        title="Validation precision–recall tradeoff",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true: pd.Series, probabilities: np.ndarray, output_path: Path) -> None:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(predicted, observed, marker="o", linewidth=2, color="#2364AA", label="Final model")
    ax.set(
        title="Test-set calibration",
        xlabel="Mean predicted failure probability",
        ylabel="Observed failure rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def permutation_importance_table(
    model,
    test: pd.DataFrame,
    *,
    sample_size: int = 5_000,
) -> pd.DataFrame:
    sample = test.sample(min(sample_size, len(test)), random_state=42)
    result = permutation_importance(
        model,
        sample[MODEL_FEATURES],
        sample[TARGET],
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def plot_importance(importance: pd.DataFrame, output_path: Path) -> None:
    shown = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.barh(shown["feature"], shown["importance_mean"], color="#3DA35D")
    ax.set(title="What changes test-set ranking performance?", xlabel="Decrease in average precision")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def segment_error_table(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    scored = test[[TARGET, "inspection_group", "risk_group", "has_history"]].copy()
    scored["predicted"] = probabilities >= threshold
    scored["false_negative"] = (scored[TARGET] == 1) & (~scored["predicted"])
    scored["false_positive"] = (scored[TARGET] == 0) & scored["predicted"]
    rows: list[dict[str, object]] = []
    for column in ("inspection_group", "risk_group", "has_history"):
        for value, group in scored.groupby(column, dropna=False):
            rows.append(
                {
                    "segment": column,
                    "value": str(value),
                    "rows": len(group),
                    "failure_rate": group[TARGET].mean(),
                    "false_negative_rate": group["false_negative"].sum()
                    / max(1, group[TARGET].sum()),
                    "false_positive_rate": group["false_positive"].sum()
                    / max(1, (group[TARGET] == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, float_format="%.5f")
