"""Reproducible training workflow from source extract to evaluated artifact."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

from inspection_risk.data import load_inspections
from inspection_risk.eda import create_eda_outputs
from inspection_risk.evaluation import (
    permutation_importance_table,
    plot_calibration,
    plot_importance,
    plot_model_comparison,
    segment_error_table,
    write_table,
)
from inspection_risk.features import MODEL_FEATURES, TARGET, chronological_split, prepare_model_frame
from inspection_risk.modeling import (
    candidate_models,
    evaluate_probabilities,
    expanding_window_scores,
    fit_and_compare,
    select_f1_threshold,
)


def _rounded(payload: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in payload.items()}


def train_and_evaluate(data_path: Path, artifact_dir: Path) -> dict[str, object]:
    frame = prepare_model_frame(load_inspections(data_path))
    split = chronological_split(frame)
    figures = artifact_dir / "figures"
    reports = artifact_dir / "reports"
    models_dir = artifact_dir / "models"
    for directory in (figures, reports, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    eda_summary = create_eda_outputs(frame, figures)
    _, validation_evaluations, validation_probabilities = fit_and_compare(
        split.train, split.validation
    )
    plot_model_comparison(
        split.validation, validation_probabilities, figures / "model_comparison.png"
    )

    selectable = {name: metrics for name, metrics in validation_evaluations.items() if "baseline" not in name}
    selected_name = max(selectable, key=lambda name: selectable[name].average_precision)
    models = candidate_models()
    cv_scores = {
        name: expanding_window_scores(split.train, models[name])
        for name in ("logistic_regression", "hist_gradient_boosting")
    }

    threshold = select_f1_threshold(
        split.validation[TARGET], validation_probabilities[selected_name]
    )
    development = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = clone(models[selected_name])
    final_model.fit(development[MODEL_FEATURES], development[TARGET])
    test_probabilities = final_model.predict_proba(split.test[MODEL_FEATURES])[:, 1]
    test_metrics = evaluate_probabilities(split.test[TARGET], test_probabilities)
    test_labels = test_probabilities >= threshold
    threshold_metrics = {
        "threshold": threshold,
        "precision": precision_score(split.test[TARGET], test_labels, zero_division=0),
        "recall": recall_score(split.test[TARGET], test_labels, zero_division=0),
        "f1": f1_score(split.test[TARGET], test_labels, zero_division=0),
        "flagged_share": float(np.mean(test_labels)),
    }

    importance = permutation_importance_table(final_model, split.test)
    errors = segment_error_table(split.test, test_probabilities, threshold=threshold)
    write_table(importance, reports / "permutation_importance.csv")
    write_table(errors, reports / "segment_error_analysis.csv")
    plot_importance(importance, figures / "feature_importance.png")
    plot_calibration(split.test[TARGET], test_probabilities, figures / "calibration.png")
    joblib.dump(final_model, models_dir / "inspection_risk.joblib")

    results: dict[str, object] = {
        "dataset": eda_summary,
        "split": {
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
            "train_end": split.train_end.date().isoformat(),
            "validation_end": split.validation_end.date().isoformat(),
            "test_end": split.test["inspection_date"].max().date().isoformat(),
            "train_failure_rate": round(float(split.train[TARGET].mean()), 4),
            "validation_failure_rate": round(float(split.validation[TARGET].mean()), 4),
            "test_failure_rate": round(float(split.test[TARGET].mean()), 4),
        },
        "validation": {
            name: _rounded(asdict(metrics)) for name, metrics in validation_evaluations.items()
        },
        "expanding_window_cv_average_precision": {
            name: {
                "folds": [round(score, 4) for score in scores],
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
            }
            for name, scores in cv_scores.items()
        },
        "selected_model": selected_name,
        "test": _rounded(asdict(test_metrics)),
        "test_at_validation_selected_threshold": _rounded(threshold_metrics),
        "interpretation_method": "test-set permutation importance scored by average precision",
    }
    (reports / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results

