from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def closest_to_value(df: pd.DataFrame, column: str, value: float) -> pd.Series:
    idx = (df[column] - value).abs().idxmin()
    return df.loc[idx]


def normalize_series(series: pd.Series) -> pd.Series:
    minimum = series.min()
    maximum = series.max()

    if maximum <= minimum:
        return series * 0.0

    return (series - minimum) / (maximum - minimum)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    df = pd.read_csv(metrics_path)

    required = {
        "sample",
        "roi_name",
        "patch_index",
        "input_times",
        "target_time",
        "average_input_coverage",
        "target_coverage",
        "union_coverage",
        "mae",
        "psnr",
        "ssim",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in metrics.csv: {missing}")

    # Prefering samples with low cloud coverage
    candidates = df[df["target_coverage"] <= 0.30].copy()

    if len(candidates) < 3:
        candidates = df.copy()

    # Scor de dificultate:
    # - union_coverage: how much of input is covered by clouds (more clouds = more difficult);
    # - average_input_coverage: how much of each input is covered by clouds;
    # - mae: if two samples have similar cloud coverage, the one with higher error is more difficult.
    candidates["difficulty_score"] = (
        0.55 * normalize_series(candidates["union_coverage"])
        + 0.25 * normalize_series(candidates["average_input_coverage"])
        + 0.20 * normalize_series(candidates["mae"])
    )

    easy = candidates.sort_values("difficulty_score", ascending=True).iloc[0]

    medium_value = candidates["difficulty_score"].median()
    medium = closest_to_value(candidates, "difficulty_score", medium_value)

    hard = candidates.sort_values("difficulty_score", ascending=False).iloc[0]

    selected = pd.DataFrame(
        [
            {"difficulty": "easy", **easy.to_dict()},
            {"difficulty": "medium", **medium.to_dict()},
            {"difficulty": "hard", **hard.to_dict()},
        ]
    )

    cols = [
        "difficulty",
        "sample",
        "roi_name",
        "patch_index",
        "input_times",
        "target_time",
        "average_input_coverage",
        "target_coverage",
        "union_coverage",
        "mae",
        "psnr",
        "ssim",
        "difficulty_score",
    ]

    selected = selected[cols]

    print()
    print(selected.to_string(index=False))
    print()

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_path, index=False)
        print(f"Saved selected samples to: {out_path}")


if __name__ == "__main__":
    main()