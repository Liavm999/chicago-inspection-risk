"""Command-line interface for acquisition and reproducible analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspection_risk.data import configured_data_path, download_inspections
from inspection_risk.training import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chicago food-inspection risk analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="fetch a fresh public-data extract")
    download.add_argument("--max-rows", type=int, default=60_000)
    download.add_argument("--since", default="2022-01-01T00:00:00.000")
    subparsers.add_parser("train", help="rebuild EDA, models, evaluation, and reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    data_path = configured_data_path(root)
    if args.command == "download":
        rows = download_inspections(data_path, since=args.since, max_rows=args.max_rows)
        print(json.dumps({"path": str(data_path), "rows": rows}, indent=2))
        return
    results = train_and_evaluate(data_path, root / "artifacts")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

