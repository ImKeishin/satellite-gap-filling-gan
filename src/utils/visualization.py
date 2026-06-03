from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def s2_rgb(x: np.ndarray) -> np.ndarray:
    """Convert (13,H,W) normalized [0,1] Sentinel-2 tensor to display RGB."""
    if x.ndim != 3 or x.shape[0] < 4:
        raise ValueError(f"Expected (C,H,W) with Sentinel-2 bands, got {x.shape}")
    rgb = np.stack([x[3], x[2], x[1]], axis=-1)  # B4, B3, B2
    return np.clip(rgb * 2.5, 0.0, 1.0)

def save_temporal_result(
        condition: np.ndarray,
        prediction: np.ndarray,
        target: np.ndarray,
        output_path: str | Path,
        n_input_times: int,
        ) -> None:
    """Save a visual comparison between cloudy input, GAN output and target."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The condition contains concatenated timestamps:
    # [time_1: 13 bands, time_2: 13 bands, ...] 
    # We visualize only the first timestamp's RGB as the "cloudy input".
    first_temporal_input = condition[:13]

    fig = plt.figure(figsize=(12, 4))

    images = [
        (first_temporal_input, "Cloudy input (first date)"),
        (prediction, "GAN reconstruction"),
        (target, "Least-cloudy target"),
    ]

    for pos, (image, title) in enumerate(
        images,
        start=1,
    ):
        ax = fig.add_subplot(1, 3, pos)
        ax.imshow(s2_rgb(image))
        ax.set_title(title)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
