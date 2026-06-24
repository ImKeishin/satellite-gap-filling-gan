from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.utils.metrics import to_zero_one
from src.utils.visualization import s2_rgb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank SEN12MS-CR-TS samples by cloud coverage."
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa")
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--cloud-detector", default="s2cloudless")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default="results_sen12mscrts/sample_ranking",
    )
    return parser.parse_args()


def save_preview(item: dict, output_path: Path, label: str, score: float) -> None:
    condition = to_zero_one(item["masked_input"], True)
    target = to_zero_one(item["target"], True)

    fig = plt.figure(figsize=(16, 4))

    for i in range(3):
        image = condition[i * 13 : (i + 1) * 13]
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(s2_rgb(image))
        ax.set_title(f"Cloudy input {i + 1}")
        ax.axis("off")

    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(s2_rgb(target))
    ax.set_title("Least-cloudy target")
    ax.axis("off")

    fig.suptitle(f"{label} case | cloud difficulty score = {score:.3f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        max_samples=args.max_samples,
    )

    rows = []

    for index in range(len(dataset)):
        item = dataset[index]

        input_times = item["input_times"].numpy()
        coverages = item["cloud_coverages"].numpy()

        selected_coverages = coverages[input_times]

        average_input_coverage = float(np.mean(selected_coverages))
        maximum_input_coverage = float(np.max(selected_coverages))
        union_mask_coverage = float(item["mask"].float().mean().item())

        # The union mask captures how much of the patch is affected across
        # the selected temporal inputs. It is used as the primary score.
        difficulty_score = union_mask_coverage

        rows.append(
            {
                "sample_index": index,
                "roi_name": item["roi_name"],
                "patch_index": int(item["patch_index"]),
                "input_times": ",".join(str(int(x)) for x in input_times),
                "target_time": int(item["target_time"]),
                "average_input_cloud_coverage": average_input_coverage,
                "maximum_input_cloud_coverage": maximum_input_coverage,
                "union_mask_coverage": union_mask_coverage,
                "difficulty_score": difficulty_score,
            }
        )

        print(
            f"\rProcessed {index + 1}/{len(dataset)} samples",
            end="",
            flush=True,
        )

    print()

    rows.sort(key=lambda row: float(row["difficulty_score"]))

    csv_path = output_dir / "sample_difficulty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    representative_positions = {
        "easy": int(0.15 * (len(rows) - 1)),
        "medium": int(0.50 * (len(rows) - 1)),
        "hard": int(0.85 * (len(rows) - 1)),
    }

    print("\nRepresentative samples:")
    for label, position in representative_positions.items():
        row = rows[position]
        sample_index = int(row["sample_index"])
        score = float(row["difficulty_score"])

        item = dataset[sample_index]
        save_preview(
            item,
            output_dir / f"{label}_sample_{sample_index:04d}.png",
            label,
            score,
        )

        print(
            f"{label:6s}: sample-index={sample_index}, "
            f"difficulty={score:.3f}, "
            f"ROI={row['roi_name']}, "
            f"patch={row['patch_index']}"
        )

    print(f"\nSaved ranking CSV and previews in: {output_dir}")


if __name__ == "__main__":
    main()