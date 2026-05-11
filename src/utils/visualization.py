from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_triplet(masked: np.ndarray, generated: np.ndarray, target: np.ndarray, output_path: str | Path, channel: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if masked.ndim == 3:
        masked = masked[channel]
        generated = generated[channel]
        target = target[channel]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input cu lipsuri", "Reconstrucție GAN", "Referință"]
    for ax, image, title in zip(axes, [masked, generated, target], titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
