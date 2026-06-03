from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.models.gan import build_models
from src.utils.metrics import mae, psnr, ssim, to_zero_one
from src.utils.visualization import save_temporal_result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SEN12MS-CR-TS temporal GAN")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints_sen12mscrts/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="all", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-visuals", type=int, default=10)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--output-dir", default="results_sen12mscrts/evaluation")
    return parser.parse_args()


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
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    generator, _ = build_models(args.n_input_times * 13, 13, args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()

    rows: list[dict[str, object]] = []
    visual_count = 0
    for batch_idx, batch in enumerate(loader):
        condition = batch["masked_input"].to(device)
        target = batch["target"].to(device)
        prediction = generator(condition)
        for item in range(condition.shape[0]):
            cond_np = to_zero_one(condition[item], True)
            pred_np = to_zero_one(prediction[item], True)
            target_np = to_zero_one(target[item], True)
            row = {
                "sample": batch_idx * args.batch_size + item,
                "roi_name": batch["roi_name"][item],
                "patch_index": int(batch["patch_index"][item]),
                "mae": mae(pred_np, target_np),
                "psnr": psnr(pred_np, target_np),
                "ssim": ssim(pred_np, target_np),
            }
            rows.append(row)
            if visual_count < args.max_visuals:
                save_temporal_result(cond_np, pred_np, target_np, output_dir / f"sample_{row['sample']:04d}.png", args.n_input_times)
                visual_count += 1

    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample", "roi_name", "patch_index", "mae", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"MAE:  {sum(float(r['mae']) for r in rows) / len(rows):.6f}")
    print(f"PSNR: {sum(float(r['psnr']) for r in rows) / len(rows):.3f}")
    print(f"SSIM: {sum(float(r['ssim']) for r in rows) / len(rows):.4f}")
    print(f"Saved metrics and visual examples in: {output_dir}")


if __name__ == "__main__":
    main()
