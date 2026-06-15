#!/usr/bin/env python3
"""
Calculează eroarea NDVI pentru Tabelul 5.5.

Rulează din rădăcina repo-ului:
    python evaluate_ndvi_sen12mscrts.py \
      --root-dir <path-to-test-root> \
      --checkpoint checkpoints_sen12mscrts_final/best.pt \
      --split test \
      --region africa \
      --max-samples 256 \
      --output-dir results_sen12mscrts/final_test_ndvi
"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NDVI error on SEN12MS-CR-TS")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints_sen12mscrts_final/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--output-dir", default="results_sen12mscrts/final_test_ndvi")
    return parser.parse_args()


def ndvi(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Ordinea benzilor este B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
    red = x[3]  # B4
    nir = x[7]  # B8
    return (nir - red) / (nir + red + eps)


@torch.no_grad()
def main() -> None:
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
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    generator, _ = build_models(args.n_input_times * 13, 13, args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()

    rows: list[dict[str, object]] = []
    for batch_idx, batch in enumerate(loader):
        condition = batch["masked_input"].to(device)
        target = batch["target"].to(device)
        prediction = generator(condition)

        for item_idx in range(condition.shape[0]):
            pred_np = to_zero_one(prediction[item_idx], True)
            target_np = to_zero_one(target[item_idx], True)
            ndvi_error = float(np.mean(np.abs(ndvi(pred_np) - ndvi(target_np))))
            rows.append(
                {
                    "sample": batch_idx * args.batch_size + item_idx,
                    "roi_name": batch["roi_name"][item_idx],
                    "patch_index": int(batch["patch_index"][item_idx]),
                    "ndvi_mae": ndvi_error,
                }
            )

    csv_path = output_dir / "ndvi_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample", "roi_name", "patch_index", "ndvi_mae"])
        writer.writeheader()
        writer.writerows(rows)

    values = np.asarray([float(row["ndvi_mae"]) for row in rows], dtype=np.float64)
    print(f"NDVI MAE mean: {values.mean():.6f}")
    print(f"NDVI MAE min:  {values.min():.6f}")
    print(f"NDVI MAE max:  {values.max():.6f}")
    print(f"Saved NDVI metrics in: {csv_path}")


if __name__ == "__main__":
    main()
