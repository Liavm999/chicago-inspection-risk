"""Model definitions, temporal comparison, and deployment-style threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from inspection_risk.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET


@dataclass(frozen=True)
class ModelEvaluation:
    average_precision: float
    roc_auc: float
    brier_score: float
    precision_at_top_10_percent: float
    recall_at_top_10_percent: float


def _logistic_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
            ),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
                ),
            ),
            ("model", LogisticRegression(max_iter=1_000, C=0.7)),
        ]
    )


def _boosted_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    min_frequency=100,
                    max_categories=200,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=180,
                    max_leaf_nodes=24,
                    min_samples_leaf=45,
                    l2_regularization=1.0,
                    categorical_features=[8, 9, 10, 11],
                    random_state=42,
                ),
            ),
        ]
    )


def candidate_models() -> dict[str, Any]:
    return {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": _logistic_pipeline(),
        "hist_gradient_boosting": _boosted_pipeline(),
    }


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> ModelEvaluation:
    # Flag exactly 10% even when a constant baseline produces tied probabilities.
    flagged = np.zeros(len(probabilities), dtype=bool)
    top_count = max(1, int(np.ceil(len(probabilities) * 0.10)))
    top_indices = np.argsort(-probabilities, kind="stable")[:top_count]
    flagged[top_indices] = True
    return ModelEvaluation(
        average_precision=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        precision_at_top_10_percent=float(precision_score(y_true, flagged, zero_division=0)),
        recall_at_top_10_percent=float(recall_score(y_true, flagged, zero_division=0)),
    )


def fit_and_compare(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, ModelEvaluation], dict[str, np.ndarray]]:
    fitted: dict[str, Any] = {}
    evaluations: dict[str, ModelEvaluation] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in candidate_models().items():
        model.fit(train[MODEL_FEATURES], train[TARGET])
        scores = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        fitted[name] = model
        probabilities[name] = scores
        evaluations[name] = evaluate_probabilities(validation[TARGET], scores)
    return fitted, evaluations, probabilities


def expanding_window_scores(frame: pd.DataFrame, model: Any, *, folds: int = 3) -> list[float]:
    """Date-grouped expanding-window AP scores; the same date never crosses a fold boundary."""
    dates = np.array(sorted(frame["inspection_date"].unique()))
    blocks = np.array_split(dates, folds + 1)
    scores = []
    for index in range(1, folds + 1):
        train_dates = np.concatenate(blocks[:index])
        validation_dates = blocks[index]
        train = frame[frame["inspection_date"].isin(train_dates)]
        validation = frame[frame["inspection_date"].isin(validation_dates)]
        fold_model = clone(model)
        fold_model.fit(train[MODEL_FEATURES], train[TARGET])
        probabilities = fold_model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        scores.append(float(average_precision_score(validation[TARGET], probabilities)))
    return scores


def select_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])
