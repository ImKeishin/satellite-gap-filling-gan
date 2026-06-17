from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.models.gan import build_models
from src.training.trainer import save_checkpoint, train_one_epoch, validate_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net + PatchGAN on SEN12MS-CR-TS Sentinel-2 time series")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument(
        "--input-selection",
        default="top_cloudy",
        choices=["top_cloudy", "random_all", "least_cloudy"],
    )
    parser.add_argument("--input-sampling-repeats", type=int, default=1)
    parser.add_argument("--min-input-cloud-coverage", type=float, default=0.05)
    parser.add_argument("--max-input-cloud-coverage", type=float, default=1.0)
    parser.add_argument("--region", default="all", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--checkpoint-dir", default="checkpoints_sen12mscrts")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    common = dict(
        root_dir=args.root_dir,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        min_input_cloud_coverage=args.min_input_cloud_coverage,
        max_input_cloud_coverage=args.max_input_cloud_coverage,
    )
    train_dataset = SEN12MSCRTSDataset(
        split="train",
        max_samples=args.max_train_samples,
        input_selection=args.input_selection,
        input_sampling_repeats=args.input_sampling_repeats,
        **common,
    )

    val_dataset = SEN12MSCRTSDataset(
        split="val",
        max_samples=args.max_val_samples,
        input_selection="least_cloudy",
        input_sampling_repeats=1,
        **common,
    )
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    generator, discriminator = build_models(
        in_channels=args.n_input_times * 13,
        out_channels=13,
        base_channels=args.base_channels,
    )
    generator.to(device)
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = {"g_loss": [], "d_loss": [], "val_l1": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        g_loss, d_loss = train_one_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, device, args.lambda_l1)
        val_l1 = validate_one_epoch(generator, val_loader, device)
        history["g_loss"].append(g_loss)
        history["d_loss"].append(d_loss)
        history["val_l1"].append(val_l1)
        print(f"Epoch {epoch:03d}/{args.epochs} | G={g_loss:.4f} | D={d_loss:.4f} | val_L1={val_l1:.4f}")
        save_checkpoint(checkpoint_dir / "last.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, history)
        if val_l1 < best_val:
            best_val = val_l1
            save_checkpoint(checkpoint_dir / "best.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, history)

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"Saved checkpoints in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
