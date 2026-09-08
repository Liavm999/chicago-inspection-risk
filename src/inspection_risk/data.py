"""Acquire and validate the public Chicago food-inspection extract."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

API_URL = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"
FIELDS = [
    "inspection_id",
    "license_",
    "facility_type",
    "risk",
    "zip",
    "inspection_date",
    "inspection_type",
    "results",
    "latitude",
    "longitude",
]
REQUIRED_COLUMNS = set(FIELDS)


def download_inspections(
    destination: Path,
    *,
    since: str = "2022-01-01T00:00:00.000",
    max_rows: int = 60_000,
    page_size: int = 10_000,
) -> int:
    """Download a narrow, chronologically ordered Socrata extract without an API token."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        while rows_written < max_rows:
            limit = min(page_size, max_rows - rows_written)
            params = {
                "$select": ",".join(FIELDS),
                "$where": (
                    f"inspection_date >= '{since}' "
                    "AND results IN ('Pass', 'Fail') AND license_ IS NOT NULL"
                ),
                "$order": "inspection_date ASC, inspection_id ASC",
                "$limit": limit,
                "$offset": rows_written,
            }
            request = Request(
                f"{API_URL}?{urlencode(params)}",
                headers={"User-Agent": "inspection-risk/0.1 (+portfolio project)"},
            )
            with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed public endpoint
                page = list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
            if not page:
                break
            writer.writerows({field: row.get(field, "") for field in FIELDS} for row in page)
            rows_written += len(page)
            if len(page) < limit:
                break
    return rows_written


def configured_data_path(project_root: Path) -> Path:
    return Path(
        os.getenv(
            "INSPECTION_DATA_PATH",
            str(project_root / "data" / "sample" / "chicago_food_inspections.csv"),
        )
    )


def load_inspections(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"inspection_id": "string", "license_": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame["inspection_date"] = pd.to_datetime(frame["inspection_date"], errors="coerce")
    return frame

