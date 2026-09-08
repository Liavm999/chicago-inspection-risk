"""Focused exploratory outputs that test assumptions used by the model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from inspection_risk.features import TARGET


def create_eda_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = (
        frame.set_index("inspection_date").resample("MS")[TARGET].agg(["mean", "size"]).reset_index()
    )
    by_type = (
        frame.groupby("inspection_group")[TARGET]
        .agg(failure_rate="mean", inspections="size")
        .query("inspections >= 50")
        .sort_values("failure_rate")
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(monthly["inspection_date"], monthly["mean"], color="#2364AA", linewidth=2)
    axes[0].set(title="Failure rate changes over time", ylabel="Failure rate", xlabel="")
    axes[0].grid(alpha=0.2)
    axes[1].barh(by_type.index, by_type["failure_rate"], color="#3DA35D")
    axes[1].set(title="Outcome differs by inspection context", xlabel="Failure rate")
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Chicago food inspections: two reasons to avoid a random split", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "rows": int(len(frame)),
        "date_min": frame["inspection_date"].min().date().isoformat(),
        "date_max": frame["inspection_date"].max().date().isoformat(),
        "failure_rate": round(float(frame[TARGET].mean()), 4),
        "licenses": int(frame["license_"].nunique()),
        "rows_with_prior_history": int(frame["has_history"].sum()),
        "failure_rate_by_inspection_group": {
            key: round(float(value), 4) for key, value in by_type["failure_rate"].items()
        },
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
