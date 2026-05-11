from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.dataset_loader import SentinelNPYDataset
from src.models.gan import build_models
from src.utils.metrics import mae, psnr, ssim, to_zero_one
from src.utils.visualization import save_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained generator")
    parser.add_argument("--root-dir", default="data/raw/S2_Spectral_Bands")
    parser.add_argument("--bands", nargs="+", default=["B01_SR"])
    parser.add_argument("--rois", nargs="*", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--max-visuals", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SentinelNPYDataset(
        root_dir=args.root_dir,
        bands=args.bands,
        rois=args.rois,
        normalize="minus1_1",
        create_synthetic_mask=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generator, _ = build_models(in_channels=len(args.bands), out_channels=len(args.bands), base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    generator.eval()

    rows = []
    visual_count = 0
    for idx, batch in enumerate(loader):
        masked_input = batch["masked_input"].to(device)
        target = batch["target"].to(device)
        generated = generator(masked_input)

        for item in range(masked_input.shape[0]):
            pred_np = to_zero_one(generated[item], normalized_minus1_1=True)
            target_np = to_zero_one(target[item], normalized_minus1_1=True)
            masked_np = to_zero_one(masked_input[item], normalized_minus1_1=True)

            row = {
                "sample": idx * args.batch_size + item,
                "roi_name": batch["roi_name"][item],
                "t": int(batch["t"][item]),
                "mae": mae(pred_np, target_np),
                "psnr": psnr(pred_np, target_np),
                "ssim": ssim(pred_np, target_np),
            }
            rows.append(row)

            if visual_count < args.max_visuals:
                save_triplet(masked_np, pred_np, target_np, output_dir / f"sample_{row['sample']:04d}.png")
                visual_count += 1

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "roi_name", "t", "mae", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    avg_mae = sum(r["mae"] for r in rows) / len(rows)
    avg_psnr = sum(r["psnr"] for r in rows) / len(rows)
    avg_ssim = sum(r["ssim"] for r in rows) / len(rows)
    print(f"MAE:  {avg_mae:.6f}")
    print(f"PSNR: {avg_psnr:.3f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"Saved metrics to: {csv_path}")


if __name__ == "__main__":
    main()
