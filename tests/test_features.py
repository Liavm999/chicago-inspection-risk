from __future__ import annotations

import pandas as pd

from inspection_risk.features import MODEL_FEATURES, chronological_split, prepare_model_frame
from inspection_risk.modeling import candidate_models


def _raw_frame() -> pd.DataFrame:
    rows = []
    for index, date in enumerate(pd.date_range("2024-01-01", periods=20, freq="D")):
        rows.append(
            {
                "inspection_id": str(index),
                "license_": "A" if index < 3 else str(index),
                "facility_type": "Restaurant",
                "risk": "Risk 1 (High)",
                "zip": "60601",
                "inspection_date": date,
                "inspection_type": "Canvass" if index % 2 else "Complaint",
                "results": "Fail" if index % 3 == 0 else "Pass",
                "latitude": 41.88,
                "longitude": -87.63,
            }
        )
    # A second same-day outcome for A must not enter the first row's history.
    rows.insert(1, {**rows[0], "inspection_id": "same-day", "results": "Pass"})
    return pd.DataFrame(rows)


def test_same_day_outcomes_do_not_leak_into_history() -> None:
    featured = prepare_model_frame(_raw_frame())
    first_day = featured[
        (featured["license_"] == "A")
        & (featured["inspection_date"] == pd.Timestamp("2024-01-01"))
    ]
    next_day = featured[
        (featured["license_"] == "A")
        & (featured["inspection_date"] == pd.Timestamp("2024-01-02"))
    ]
    assert (first_day["prior_inspections"] == 0).all()
    assert (first_day["prior_failures"] == 0).all()
    assert next_day.iloc[0]["prior_inspections"] == 2
    assert next_day.iloc[0]["prior_failures"] == 1


def test_chronological_split_keeps_dates_disjoint() -> None:
    split = chronological_split(prepare_model_frame(_raw_frame()))
    assert split.train["inspection_date"].max() < split.validation["inspection_date"].min()
    assert split.validation["inspection_date"].max() < split.test["inspection_date"].min()


def test_pipeline_accepts_unseen_categories() -> None:
    featured = prepare_model_frame(_raw_frame())
    model = candidate_models()["logistic_regression"]
    model.fit(featured[MODEL_FEATURES], featured["failed"])
    unseen = featured.tail(1).copy()
    unseen["facility_group"] = "Never seen facility"
    probabilities = model.predict_proba(unseen[MODEL_FEATURES])[:, 1]
    assert 0 <= probabilities[0] <= 1

