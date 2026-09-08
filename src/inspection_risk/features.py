"""Leakage-safe cleaning and feature engineering for longitudinal inspections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "failed"
NUMERIC_FEATURES = [
    "prior_inspections",
    "prior_failures",
    "smoothed_prior_fail_rate",
    "days_since_previous",
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
]
CATEGORICAL_FEATURES = ["facility_group", "risk_group", "inspection_group", "zip"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def _collapse_rare(series: pd.Series, *, minimum_count: int) -> pd.Series:
    cleaned = series.fillna("Unknown").astype(str).str.strip()
    common = cleaned.value_counts()[lambda counts: counts >= minimum_count].index
    return cleaned.where(cleaned.isin(common), "Other")


def _inspection_group(series: pd.Series) -> pd.Series:
    normalized = series.fillna("Unknown").astype(str).str.lower()
    output = pd.Series("Other", index=series.index, dtype="string")
    output[normalized.str.contains("canvass")] = "Canvass"
    output[normalized.str.contains("complaint|food poisoning")] = "Complaint"
    output[normalized.str.contains("license")] = "License"
    output[normalized.str.contains("re-inspection|reinspection")] = "Re-inspection"
    return output


def prepare_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create only features that would be known immediately before each inspection.

    History is calculated at a license/date grain so another inspection from the same
    calendar day can never leak its result into a peer row.
    """
    frame = raw.copy()
    frame = frame[
        frame["results"].isin(["Pass", "Fail"])
        & frame["license_"].notna()
        & frame["inspection_date"].notna()
    ].copy()
    frame = frame.drop_duplicates("inspection_id", keep="last")
    frame = frame.sort_values(["inspection_date", "inspection_id"], kind="stable")
    frame[TARGET] = (frame["results"] == "Fail").astype("int8")

    daily = (
        frame.groupby(["license_", "inspection_date"], as_index=False, sort=True)[TARGET]
        .agg(day_inspections="size", day_failures="sum")
        .sort_values(["license_", "inspection_date"], kind="stable")
    )
    by_license = daily.groupby("license_", sort=False)
    daily["prior_inspections"] = (
        by_license["day_inspections"].cumsum() - daily["day_inspections"]
    )
    daily["prior_failures"] = by_license["day_failures"].cumsum() - daily["day_failures"]
    daily["previous_date"] = by_license["inspection_date"].shift(1)
    daily["days_since_previous"] = (
        daily["inspection_date"] - daily["previous_date"]
    ).dt.days.clip(lower=0, upper=1_095)
    history = daily[
        [
            "license_",
            "inspection_date",
            "prior_inspections",
            "prior_failures",
            "days_since_previous",
        ]
    ]
    frame = frame.merge(history, on=["license_", "inspection_date"], how="left", validate="m:1")

    # A fixed weak Beta prior avoids unstable 0/1 rates after a single visit.
    frame["smoothed_prior_fail_rate"] = (
        frame["prior_failures"] + 1.0
    ) / (frame["prior_inspections"] + 4.0)
    frame["facility_group"] = _collapse_rare(frame["facility_type"], minimum_count=100)
    frame["risk_group"] = (
        frame["risk"].fillna("Unknown").str.extract(r"(Risk \d)", expand=False).fillna("Unknown")
    )
    frame["inspection_group"] = _inspection_group(frame["inspection_type"])
    frame["zip"] = frame["zip"].fillna("Unknown").astype(str).str.replace(r"\.0$", "", regex=True)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    radians = 2 * np.pi * frame["inspection_date"].dt.month / 12
    frame["month_sin"] = np.sin(radians)
    frame["month_cos"] = np.cos(radians)
    frame["has_history"] = frame["prior_inspections"] > 0
    return frame.reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """Split whole calendar dates to mimic deployment into genuinely later periods."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test set")

    ordered = frame.sort_values(["inspection_date", "inspection_id"], kind="stable")
    dates = ordered["inspection_date"].sort_values().unique()
    train_index = max(0, min(len(dates) - 3, int(len(dates) * train_fraction) - 1))
    validation_index = max(
        train_index + 1,
        min(len(dates) - 2, int(len(dates) * (train_fraction + validation_fraction)) - 1),
    )
    train_end = pd.Timestamp(dates[train_index])
    validation_end = pd.Timestamp(dates[validation_index])
    train = ordered[ordered["inspection_date"] <= train_end].copy()
    validation = ordered[
        (ordered["inspection_date"] > train_end)
        & (ordered["inspection_date"] <= validation_end)
    ].copy()
    test = ordered[ordered["inspection_date"] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("temporal split produced an empty partition")
    return TemporalSplit(train, validation, test, train_end, validation_end)

