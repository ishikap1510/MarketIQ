"""
export_csvs.py — Export cleaned and feature-engineered DataFrames to CSV.

Run this script directly to write all three processed datasets to data/processed/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import (
    load_city_data,
    load_cert_data,
    load_skill_data,
    compute_city_features,
    compute_cert_features,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def export_all(output_dir: str = OUTPUT_DIR) -> None:
    """Load, process, and export all three datasets to CSV.

    Args:
        output_dir: Directory path where CSVs will be written.

    Returns:
        None — files written to disk, paths printed to stdout.
    """
    os.makedirs(output_dir, exist_ok=True)

    exports = {
        "city_metrics.csv":  compute_city_features(load_city_data()),
        "cert_roi.csv":      compute_cert_features(load_cert_data()),
        "skill_demand.csv":  load_skill_data(),
    }

    for filename, df in exports.items():
        path = os.path.join(output_dir, filename)
        df.to_csv(path, index=False)
        print(f"  Exported {len(df)} rows → {path}")

    print(f"\nAll {len(exports)} files written to: {output_dir}")


if __name__ == "__main__":
    print("Exporting processed datasets...\n")
    export_all()
