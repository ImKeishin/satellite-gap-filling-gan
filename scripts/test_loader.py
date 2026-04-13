import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.dataset.dataset_loader import SentinelNPYDataset


def main():
    print("START TEST LOADER")

    dataset = SentinelNPYDataset(
        root_dir="data/raw/S2_Spectral_Bands",
        bands=["B01_SR"],
        rois=["roi1"],
        normalize="zero_one",
        create_synthetic_mask=True,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    x = sample["masked_input"].numpy()
    y = sample["target"].numpy()
    m = sample["mask"].numpy()[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(x[0], cmap="gray")
    axes[0].set_title("Masked input")
    axes[0].axis("off")

    axes[1].imshow(y[0], cmap="gray")
    axes[1].set_title("Target")
    axes[1].axis("off")

    axes[2].imshow(m, cmap="gray")
    axes[2].set_title("Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("results/test_loader_output.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved figure to: results/test_loader_output.png")
    print("ROI:", sample["roi_name"])
    print("Time index:", sample["t"])


if __name__ == "__main__":
    main()