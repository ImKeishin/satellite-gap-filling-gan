from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset.dataset_loader import SentinelNPYDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual test for SentinelNPYDataset")
    parser.add_argument("--root-dir", default="data/raw/S2_Spectral_Bands")
    parser.add_argument("--bands", nargs="+", default=["B01_SR"])
    parser.add_argument("--rois", nargs="*", default=None)
    parser.add_argument("--output", default="results/test_loader_output.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SentinelNPYDataset(
        root_dir=args.root_dir,
        bands=args.bands,
        rois=args.rois,
        normalize="zero_one",
        create_synthetic_mask=True,
    )
    print("Dataset size:", len(dataset))
    sample = dataset[0]
    x = sample["masked_input"].numpy()
    y = sample["target"].numpy()
    m = sample["mask"].numpy()[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Masked input")
    axes[0].axis("off")
    axes[1].imshow(y[0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Target")
    axes[1].axis("off")
    axes[2].imshow(m, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure to:", output)
    print("ROI:", sample["roi_name"])
    print("Time index:", sample["t"])


if __name__ == "__main__":
    main()
