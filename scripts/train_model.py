from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.dataset_loader import SentinelNPYDataset
from src.models.gan import build_models
from src.training.trainer import save_checkpoint, train_one_epoch, validate_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN for Sentinel-2 gap filling")
    parser.add_argument("--root-dir", default="data/raw/S2_Spectral_Bands")
    parser.add_argument("--bands", nargs="+", default=["B01_SR"])
    parser.add_argument("--rois", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--normalize", default="minus1_1", choices=["none", "zero_one", "minus1_1"])
    parser.add_argument("--preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SentinelNPYDataset(
        root_dir=args.root_dir,
        bands=args.bands,
        rois=args.rois,
        normalize=args.normalize,
        create_synthetic_mask=True,
        preload=args.preload,
    )
    print(f"Dataset size: {len(dataset)} samples")

    val_size = max(1, int(len(dataset) * args.val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    in_channels = len(args.bands)
    generator, discriminator = build_models(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
    )
    generator.to(device)
    discriminator.to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history = {"g_loss": [], "d_loss": [], "val_l1": []}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        g_loss, d_loss = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            device=device,
            lambda_l1=args.lambda_l1,
        )
        val_l1 = validate_one_epoch(generator, val_loader, device) if val_loader is not None else None

        history["g_loss"].append(g_loss)
        history["d_loss"].append(d_loss)
        history["val_l1"].append(val_l1)

        msg = f"Epoch {epoch:03d}/{args.epochs} | G: {g_loss:.4f} | D: {d_loss:.4f}"
        if val_l1 is not None:
            msg += f" | Val L1: {val_l1:.4f}"
        print(msg)

        save_checkpoint(
            checkpoint_dir / "last.pt",
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            epoch,
            history,
        )
        if epoch == 1 or val_l1 == min([v for v in history["val_l1"] if v is not None]):
            save_checkpoint(checkpoint_dir / "best.pt", generator, discriminator, g_optimizer, d_optimizer, epoch, history)

    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoints to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
