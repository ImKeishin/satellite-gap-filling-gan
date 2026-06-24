from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.models.gan import build_models
from src.utils.metrics import to_zero_one


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NDVI MAE for SEN12MS-CR-TS temporal GAN")

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa")
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=256)

    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--input-selection", default="random_all", choices=["top_cloudy", "random_all", "least_cloudy"])
    parser.add_argument("--input-random-seed", type=int, default=42)
    parser.add_argument("--min-input-cloud-coverage", type=float, default=0.05)
    parser.add_argument("--max-input-cloud-coverage", type=float, default=0.95)

    parser.add_argument("--output-dir", default="results_sen12mscrts/final_test_random_all_r3")

    return parser.parse_args()


def compute_ndvi(image_13bands: np.ndarray) -> np.ndarray:
    # Ordinea benzilor Sentinel-2: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
    red = image_13bands[3]
    nir = image_13bands[7]
    return (nir - red) / (nir + red + 1e-6)


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        max_samples=args.max_samples,
        cloud_detector=args.cloud_detector,
        input_selection=args.input_selection,
        input_sampling_repeats=1,
        random_seed=args.input_random_seed,
        min_input_cloud_coverage=args.min_input_cloud_coverage,
        max_input_cloud_coverage=args.max_input_cloud_coverage,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    generator, _ = build_models(
        args.n_input_times * 13,
        13,
        args.base_channels,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()

    rows = []

    for batch_idx, batch in enumerate(loader):
        condition = batch["masked_input"].to(device)
        target = batch["target"].to(device)
        prediction = generator(condition)

        for item in range(condition.shape[0]):
            pred_np = to_zero_one(prediction[item], True)
            target_np = to_zero_one(target[item], True)

            pred_ndvi = compute_ndvi(pred_np)
            target_ndvi = compute_ndvi(target_np)

            mae_ndvi = float(np.mean(np.abs(pred_ndvi - target_ndvi)))

            sample_index = batch_idx * args.batch_size + item

            rows.append(
                {
                    "sample": sample_index,
                    "mae_ndvi": mae_ndvi,
                }
            )

    values = np.array([row["mae_ndvi"] for row in rows], dtype=np.float64)

    csv_path = output_dir / "ndvi_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample", "mae_ndvi"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"MAE_NDVI mediu: {values.mean():.6f}")
    print(f"MAE_NDVI minim: {values.min():.6f}")
    print(f"MAE_NDVI maxim: {values.max():.6f}")
    print(f"Saved NDVI metrics in: {csv_path}")


if __name__ == "__main__":
    main()