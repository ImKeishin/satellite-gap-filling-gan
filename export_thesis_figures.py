#!/usr/bin/env python3
"""
Exportă figurile reproductibile pentru lucrare.

Rulează din rădăcina repo-ului:
    python export_thesis_figures.py \
      --root-dir <path-to-test-root> \
      --history checkpoints_sen12mscrts_final/history.json \
      --sample-index 0 \
      --cloud-detector s2cloudless \
      --output-dir thesis_figures

Fără --root-dir sunt generate diagramele conceptuale și graficele din history.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.utils.metrics import to_zero_one
from src.utils.visualization import s2_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis figures")
    parser.add_argument("--root-dir", default=None, help="Folderul care conține direct directoarele ROIs...")
    parser.add_argument("--history", default="checkpoints_sen12mscrts_final/history.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--output-dir", default="thesis_figures")
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def add_block(ax, x: float, y: float, width: float, height: float, text: str, fontsize: int = 10) -> None:
    ax.add_patch(Rectangle((x, y), width, height, fill=False, linewidth=1.5))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def export_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    add_block(ax, 0.3, 3.8, 2.3, 1.1, "30 observații\nSentinel-2")
    add_block(ax, 3.1, 3.8, 2.3, 1.1, "Măști de nori\ns2cloudless")
    add_block(ax, 5.9, 4.4, 2.6, 1.0, "Țintă:\nacoperire minimă")
    add_block(ax, 5.9, 2.7, 2.6, 1.0, "3 intrări:\nobservații mai afectate")
    add_block(ax, 9.0, 2.7, 2.3, 1.0, "Concatenare\n39 canale")
    add_block(ax, 11.9, 2.7, 1.8, 1.0, "Generator\nU-Net")
    add_block(ax, 14.2, 2.7, 1.5, 1.0, "Predicție\n13 benzi")
    add_block(ax, 11.9, 0.6, 2.0, 1.0, "Discriminator\nPatchGAN")
    add_block(ax, 9.0, 0.6, 2.3, 1.0, "Pereche reală\nsau generată")

    add_arrow(ax, (2.6, 4.35), (3.1, 4.35))
    add_arrow(ax, (5.4, 4.35), (5.9, 4.9))
    add_arrow(ax, (5.4, 4.05), (5.9, 3.2))
    add_arrow(ax, (8.5, 3.2), (9.0, 3.2))
    add_arrow(ax, (11.3, 3.2), (11.9, 3.2))
    add_arrow(ax, (13.7, 3.2), (14.2, 3.2))
    add_arrow(ax, (15.0, 2.7), (13.9, 1.1), rad=-0.15)
    add_arrow(ax, (8.5, 4.9), (10.1, 1.6), rad=0.18)
    add_arrow(ax, (11.3, 1.1), (11.9, 1.1))
    ax.set_title("Fluxul general al metodologiei propuse", fontsize=14)
    save(fig, output_dir / "fig_4_1_flux_metodologie.png")


def export_unet(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    encoder = [
        (0.4, 5.2, "Intrare\n39 canale"),
        (2.3, 4.4, "Encoder 1"),
        (4.2, 3.6, "Encoder 2"),
        (6.1, 2.8, "Bottleneck"),
    ]
    decoder = [
        (8.2, 3.6, "Decoder 1"),
        (10.1, 4.4, "Decoder 2"),
        (12.0, 5.2, "Ieșire\n13 canale"),
    ]

    for x, y, label in encoder + decoder:
        add_block(ax, x, y, 1.45, 0.9, label, fontsize=9)

    points = [(1.85, 5.65), (2.3, 4.85), (3.75, 4.85), (4.2, 4.05), (5.65, 4.05), (6.1, 3.25),
              (7.55, 3.25), (8.2, 4.05), (9.65, 4.05), (10.1, 4.85), (11.55, 4.85), (12.0, 5.65)]
    for start, end in zip(points[::2], points[1::2]):
        add_arrow(ax, start, end)

    add_arrow(ax, (3.0, 5.3), (10.8, 5.3), rad=-0.28)
    add_arrow(ax, (4.9, 4.5), (8.9, 4.5), rad=-0.22)

    ax.text(7.3, 6.15, "skip connections", ha="center", va="center", fontsize=10)
    ax.set_title("Generator U-Net: encoder, decoder și conexiuni directe", fontsize=14)
    save(fig, output_dir / "fig_4_3_generator_unet.png")


def export_patchgan(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    add_block(ax, 0.4, 2.4, 2.0, 0.9, "Condiție temporală\n39 canale")
    add_block(ax, 0.4, 0.7, 2.0, 0.9, "Țintă reală sau\npredicție: 13 canale")
    add_block(ax, 3.2, 1.55, 1.8, 0.9, "Concatenare\n52 canale")
    add_block(ax, 5.8, 1.55, 1.7, 0.9, "Blocuri\nconvoluționale")
    add_block(ax, 8.3, 1.55, 1.7, 0.9, "PatchGAN")
    add_block(ax, 10.8, 1.55, 2.4, 0.9, "Hartă de scoruri\nlocale")

    add_arrow(ax, (2.4, 2.85), (3.2, 2.1))
    add_arrow(ax, (2.4, 1.15), (3.2, 1.9))
    add_arrow(ax, (5.0, 2.0), (5.8, 2.0))
    add_arrow(ax, (7.5, 2.0), (8.3, 2.0))
    add_arrow(ax, (10.0, 2.0), (10.8, 2.0))

    ax.set_title("Discriminator condiționat PatchGAN", fontsize=14)
    save(fig, output_dir / "fig_4_4_discriminator_patchgan.png")


def export_history(history_path: Path, output_dir: Path) -> None:
    if not history_path.exists():
        print(f"History not found, skipped plots: {history_path}")
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    plots = [
        ("g_loss", "Pierderea generatorului", "Pierdere", "fig_5_1_pierderea_generatorului.png"),
        ("d_loss", "Pierderea discriminatorului", "Pierdere", "fig_5_2_pierderea_discriminatorului.png"),
        ("val_l1", "Eroarea L1 pe setul de validare", "Eroare L1", "fig_5_3_l1_validare.png"),
    ]

    for key, title, ylabel, filename in plots:
        values = history[key]
        epochs = np.arange(1, len(values) + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, values, marker="o")
        ax.set_xlabel("Epoca")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if key == "val_l1":
            best_idx = int(np.argmin(values))
            ax.scatter([epochs[best_idx]], [values[best_idx]], zorder=3)
            ax.annotate(
                f"minim: {values[best_idx]:.6f}, epoca {epochs[best_idx]}",
                (epochs[best_idx], values[best_idx]),
                xytext=(8, 10),
                textcoords="offset points",
            )
        save(fig, output_dir / filename)


def export_data_figures(args: argparse.Namespace, output_dir: Path) -> None:
    if args.root_dir is None:
        print("No --root-dir supplied, skipped figures 2.1, 2.2 and 4.2.")
        return

    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
    )
    item = dataset[args.sample_index]
    condition = to_zero_one(item["masked_input"], True)
    target = to_zero_one(item["target"], True)
    input_times = [int(x) for x in item["input_times"].tolist()]
    union_mask = item["mask"].detach().cpu().numpy()[0]

    sample = dataset.samples[args.sample_index]
    raw_sequence = [dataset._read_s2(path) for path in sample.s2_paths]
    all_masks = [dataset._cloud_mask(image) for image in raw_sequence]
    selected_masks = [all_masks[t] for t in input_times]

    # Figura 2.1
    fig = plt.figure(figsize=(16, 4))
    for pos in range(args.n_input_times):
        ax = fig.add_subplot(1, args.n_input_times + 1, pos + 1)
        image = condition[pos * 13 : (pos + 1) * 13]
        ax.imshow(s2_rgb(image))
        ax.set_title(f"Intrare t={input_times[pos]}")
        ax.axis("off")
    ax = fig.add_subplot(1, args.n_input_times + 1, args.n_input_times + 1)
    ax.imshow(s2_rgb(target))
    ax.set_title(f"Țintă t={int(item['target_time'])}")
    ax.axis("off")
    fig.tight_layout()
    save(fig, output_dir / "fig_2_1_esantion_temporal.png")

    # Figura 2.2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(s2_rgb(condition[:13]))
    axes[0].set_title("Imagine afectată de nori")
    axes[1].imshow(union_mask)
    axes[1].set_title("Mască agregată")
    axes[2].imshow(s2_rgb(target))
    axes[2].set_title("Imagine-țintă")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save(fig, output_dir / "fig_2_2_intrare_masca_tinta.png")

    # Figura 4.2
    fig = plt.figure(figsize=(16, 4))
    for pos, mask in enumerate(selected_masks, start=1):
        ax = fig.add_subplot(1, args.n_input_times + 1, pos)
        ax.imshow(mask)
        ax.set_title(f"Mască t={input_times[pos - 1]}")
        ax.axis("off")
    ax = fig.add_subplot(1, args.n_input_times + 1, args.n_input_times + 1)
    ax.imshow(union_mask)
    ax.set_title("Reuniunea măștilor")
    ax.axis("off")
    fig.tight_layout()
    save(fig, output_dir / "fig_4_2_masca_agregata.png")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_pipeline(output_dir)
    export_unet(output_dir)
    export_patchgan(output_dir)
    export_history(Path(args.history), output_dir)
    export_data_figures(args, output_dir)


if __name__ == "__main__":
    main()
