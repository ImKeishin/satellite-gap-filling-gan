#!/usr/bin/env python3
"""Generate thesis figures and auxiliary values for satellite-gap-filling-gan.

Run this file from the repository root, for example:
    python thesis_assets.py history
    python thesis_assets.py concepts
    python thesis_assets.py dataset --root-dir D:\\data\\sen12mscrts\\test --sample-index 38
    python thesis_assets.py count --train-root D:\\data\\sen12mscrts\\train --test-root D:\\data\\sen12mscrts\\test
    python thesis_assets.py ndvi --root-dir D:\\data\\sen12mscrts\\test --checkpoint checkpoints_sen12mscrts_final\\best.pt
    python thesis_assets.py rank-overlap --root-dir D:\\data\\sen12mscrts\\test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_line_plot(values: Iterable[float], ylabel: str, title: str, output: Path) -> None:
    values = list(values)
    epochs = np.arange(1, len(values) + 1)
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, values, marker="o")
    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def cmd_history(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    save_line_plot(history["g_loss"], "Pierderea generatorului", "Evoluția pierderii generatorului", out / "fig_5_1_generator_loss.png")
    save_line_plot(history["d_loss"], "Pierderea discriminatorului", "Evoluția pierderii discriminatorului", out / "fig_5_2_discriminator_loss.png")
    save_line_plot(history["val_l1"], "Eroarea L1 pe validare", "Evoluția erorii L1 pe validare", out / "fig_5_3_validation_l1.png")
    print(f"Saved training curves in: {out}")


def _boxes(ax, labels, x_positions, y, width=0.14, height=0.12):
    from matplotlib.patches import FancyBboxPatch
    centers = []
    for label, x in zip(labels, x_positions):
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", fill=False)
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
        centers.append((x + width / 2, y + height / 2))
    return centers


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->"})


def _prepare_canvas(title: str):
    fig = plt.figure(figsize=(13, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title)
    return fig, ax


def cmd_concepts(args: argparse.Namespace) -> None:
    out = ensure_dir(args.output_dir)

    fig, ax = _prepare_canvas("Fluxul general al metodologiei propuse")
    labels = [
        "30 observații\nSentinel-2",
        "Măști de nori\ns2cloudless",
        "Țintă: observația\ncu cei mai puțini nori",
        "3 intrări temporale\n3 × 13 = 39 canale",
        "Generator\nU-Net",
        "Reconstrucție\n13 benzi",
        "Discriminator\nPatchGAN",
    ]
    xs = [0.02, 0.16, 0.31, 0.47, 0.62, 0.75, 0.87]
    centers = _boxes(ax, labels, xs, 0.45, width=0.11, height=0.18)
    for left, right in zip(centers[:-1], centers[1:]):
        _arrow(ax, (left[0] + 0.055, left[1]), (right[0] - 0.055, right[1]))
    ax.text(0.925, 0.30, "real / generat", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_4_1_pipeline.png", dpi=200)
    plt.close(fig)

    fig, ax = _prepare_canvas("Generatorul U-Net implementat")
    enc_labels = ["39×256×256", "32×128×128", "64×64×64", "128×32×32", "256×16×16", "256×8×8", "256×4×4"]
    enc_x = [0.02, 0.14, 0.26, 0.38, 0.50, 0.62, 0.74]
    enc = _boxes(ax, enc_labels, enc_x, 0.67, width=0.10, height=0.11)
    for left, right in zip(enc[:-1], enc[1:]):
        _arrow(ax, (left[0] + 0.05, left[1]), (right[0] - 0.05, right[1]))
    dec_labels = ["256×8×8", "256×16×16", "128×32×32", "64×64×64", "32×128×128", "13×256×256"]
    dec_x = [0.74, 0.62, 0.50, 0.38, 0.26, 0.14]
    dec = _boxes(ax, dec_labels, dec_x, 0.24, width=0.10, height=0.11)
    _arrow(ax, (enc[-1][0], enc[-1][1] - 0.055), (dec[0][0], dec[0][1] + 0.055))
    for left, right in zip(dec[:-1], dec[1:]):
        _arrow(ax, (left[0] - 0.05, left[1]), (right[0] + 0.05, right[1]))
    # skip connections: encoder d1..d5 to corresponding decoder levels
    for e, d in zip(enc[1:6], reversed(dec[:5])):
        ax.annotate("", xy=(d[0], d[1] + 0.055), xytext=(e[0], e[1] - 0.055), arrowprops={"arrowstyle": "->", "linestyle": "--"})
    ax.text(0.50, 0.52, "skip connections", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_4_3_unet.png", dpi=200)
    plt.close(fig)

    fig, ax = _prepare_canvas("Discriminatorul condiționat PatchGAN")
    labels = [
        "Condiție temporală\n39 canale",
        "Imagine reală sau\ngenerată: 13 canale",
        "Concatenare\n52 canale",
        "Conv 32",
        "Conv 64",
        "Conv 128",
        "Conv 256",
        "Hartă locală\nde scoruri",
    ]
    xs = [0.02, 0.16, 0.31, 0.45, 0.56, 0.67, 0.78, 0.89]
    centers = _boxes(ax, labels, xs, 0.45, width=0.09, height=0.18)
    _arrow(ax, (centers[0][0] + 0.045, centers[0][1]), (centers[2][0] - 0.045, centers[2][1]))
    _arrow(ax, (centers[1][0] + 0.045, centers[1][1]), (centers[2][0] - 0.045, centers[2][1] - 0.04))
    for left, right in zip(centers[2:-1], centers[3:]):
        _arrow(ax, (left[0] + 0.045, left[1]), (right[0] - 0.045, right[1]))
    fig.tight_layout()
    fig.savefig(out / "fig_4_4_patchgan.png", dpi=200)
    plt.close(fig)
    print(f"Saved conceptual diagrams in: {out}")


def _build_dataset(args: argparse.Namespace):
    from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
    return SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        max_samples=getattr(args, "max_samples", None),
    )


def _selected_masks(dataset, item):
    sample = dataset.samples[int(item["patch_index"]) if False else 0]  # only for type hints
    del sample
    # item comes from dataset[sample_index]; use the corresponding Sample stored by the loader.
    return None


def _item_payload(dataset, sample_index: int):
    from src.utils.metrics import to_zero_one
    from src.utils.visualization import s2_rgb
    item = dataset[sample_index]
    sample = dataset.samples[sample_index]
    input_times = [int(x) for x in item["input_times"].tolist()]
    masks = []
    for t in input_times:
        raw = dataset._read_s2(sample.s2_paths[t])
        masks.append(dataset._cloud_mask(raw))
    condition = to_zero_one(item["masked_input"], True)
    target = to_zero_one(item["target"], True)
    inputs = [condition[i * 13 : (i + 1) * 13] for i in range(len(input_times))]
    return item, input_times, masks, inputs, target, s2_rgb


def cmd_dataset(args: argparse.Namespace) -> None:
    dataset = _build_dataset(args)
    out = ensure_dir(args.output_dir)
    item, input_times, masks, inputs, target, s2_rgb = _item_payload(dataset, args.sample_index)
    union = np.maximum.reduce(masks)
    intersection = np.minimum.reduce(masks)

    fig = plt.figure(figsize=(16, 4))
    for i, image in enumerate(inputs):
        ax = fig.add_subplot(1, len(inputs) + 1, i + 1)
        ax.imshow(s2_rgb(image))
        ax.set_title(f"Intrare temporală {i + 1} (t={input_times[i]})")
        ax.axis("off")
    ax = fig.add_subplot(1, len(inputs) + 1, len(inputs) + 1)
    ax.imshow(s2_rgb(target))
    ax.set_title("Țintă cu acoperire minimă")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_2_1_temporal_sample.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(13, 4))
    panels = [
        (s2_rgb(inputs[0]), "Intrare afectată de nori"),
        (union, "Mască agregată: reuniune"),
        (s2_rgb(target), "Imagine-țintă"),
    ]
    for i, (image, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_2_2_cloud_mask_target.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(17, 4))
    all_masks = [(mask, f"Mască intrare {i + 1}") for i, mask in enumerate(masks)] + [
        (union, "Reuniune: afectat în cel puțin o intrare"),
        (intersection, "Intersecție: afectat în toate intrările"),
    ]
    for i, (image, title) in enumerate(all_masks, start=1):
        ax = fig.add_subplot(1, len(all_masks), i)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_4_2_aggregate_masks.png", dpi=200)
    plt.close(fig)

    print(f"Saved data-based figures in: {out}")
    print(f"Selected input times: {input_times}")
    print(f"Union mask coverage: {float(union.mean()):.6f}")
    print(f"Intersection mask coverage: {float(intersection.mean()):.6f}")


def cmd_count(args: argparse.Namespace) -> None:
    from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
    train = SEN12MSCRTSDataset(args.train_root, split="train", region=args.region, cloud_detector="heuristic")
    val = SEN12MSCRTSDataset(args.train_root, split="val", region=args.region, cloud_detector="heuristic")
    test = SEN12MSCRTSDataset(args.test_root, split="test", region=args.region, cloud_detector="heuristic")
    print(f"Antrenare: {len(train)}")
    print(f"Validare:  {len(val)}")
    print(f"Testare:   {len(test)}")


def cmd_ndvi(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
    from src.models.gan import build_models
    from src.utils.metrics import to_zero_one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    rows = []
    eps = 1e-6
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            condition = batch["masked_input"].to(device)
            target = batch["target"].to(device)
            prediction = generator(condition)
            for item_idx in range(condition.shape[0]):
                pred = to_zero_one(prediction[item_idx], True)
                real = to_zero_one(target[item_idx], True)
                pred_ndvi = (pred[7] - pred[3]) / (pred[7] + pred[3] + eps)  # B8, B4
                real_ndvi = (real[7] - real[3]) / (real[7] + real[3] + eps)
                value = float(np.mean(np.abs(real_ndvi - pred_ndvi)))
                rows.append({"sample": batch_idx * args.batch_size + item_idx, "mae_ndvi": value})
    out = ensure_dir(args.output_dir)
    with (out / "ndvi_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample", "mae_ndvi"])
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([row["mae_ndvi"] for row in rows], dtype=np.float64)
    print(f"MAE_NDVI mean: {values.mean():.6f}")
    print(f"MAE_NDVI min:  {values.min():.6f}")
    print(f"MAE_NDVI max:  {values.max():.6f}")
    print(f"Saved NDVI CSV in: {out}")


def cmd_rank_overlap(args: argparse.Namespace) -> None:
    from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
    from src.utils.visualization import s2_rgb
    from src.utils.metrics import to_zero_one

    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        max_samples=args.max_samples,
    )
    out = ensure_dir(args.output_dir)
    rows = []
    for sample_index in range(len(dataset)):
        item = dataset[sample_index]
        sample = dataset.samples[sample_index]
        input_times = [int(x) for x in item["input_times"].tolist()]
        masks = [dataset._cloud_mask(dataset._read_s2(sample.s2_paths[t])) for t in input_times]
        union = np.maximum.reduce(masks)
        intersection = np.minimum.reduce(masks)
        rows.append({
            "sample_index": sample_index,
            "roi_name": item["roi_name"],
            "patch_index": int(item["patch_index"]),
            "input_times": ",".join(map(str, input_times)),
            "target_time": int(item["target_time"]),
            "union_mask_coverage": float(union.mean()),
            "common_cloud_coverage": float(intersection.mean()),
            "difficulty_score": float(intersection.mean()),
        })
        print(f"\rProcessed {sample_index + 1}/{len(dataset)}", end="", flush=True)
    print()
    rows.sort(key=lambda row: row["difficulty_score"])
    with (out / "sample_difficulty_overlap.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    positions = {"easy": int(0.15 * (len(rows) - 1)), "medium": int(0.50 * (len(rows) - 1)), "hard": int(0.85 * (len(rows) - 1))}
    for label, position in positions.items():
        row = rows[position]
        sample_index = int(row["sample_index"])
        item = dataset[sample_index]
        condition = to_zero_one(item["masked_input"], True)
        target = to_zero_one(item["target"], True)
        fig = plt.figure(figsize=(16, 4))
        for i in range(args.n_input_times):
            ax = fig.add_subplot(1, args.n_input_times + 1, i + 1)
            ax.imshow(s2_rgb(condition[i * 13 : (i + 1) * 13]))
            ax.set_title(f"Intrare {i + 1}")
            ax.axis("off")
        ax = fig.add_subplot(1, args.n_input_times + 1, args.n_input_times + 1)
        ax.imshow(s2_rgb(target))
        ax.set_title("Țintă cu acoperire minimă")
        ax.axis("off")
        fig.suptitle(f"{label} | common-cloud score = {row['difficulty_score']:.3f}")
        fig.tight_layout()
        fig.savefig(out / f"{label}_sample_{sample_index:04d}.png", dpi=180)
        plt.close(fig)
        print(f"{label}: sample-index={sample_index}, difficulty={row['difficulty_score']:.6f}")
    print(f"Saved overlap ranking in: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("history", help="Generate Figures 5.1-5.3 from history.json")
    p.add_argument("--history", default="checkpoints_sen12mscrts_final/history.json")
    p.add_argument("--output-dir", default="thesis_figures")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("concepts", help="Generate conceptual diagrams for Figures 4.1, 4.3 and 4.4")
    p.add_argument("--output-dir", default="thesis_figures")
    p.set_defaults(func=cmd_concepts)

    p = sub.add_parser("dataset", help="Generate Figures 2.1, 2.2 and 4.2 from a dataset sample")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    p.add_argument("--sample-index", type=int, default=38)
    p.add_argument("--n-input-times", type=int, default=3)
    p.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", default="thesis_figures")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("count", help="Count indexed train, validation and test patches for Table 2.2")
    p.add_argument("--train-root", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("ndvi", help="Calculate NDVI reconstruction error for Table 5.5")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints_sen12mscrts_final/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    p.add_argument("--n-input-times", type=int, default=3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    p.add_argument("--output-dir", default="results_sen12mscrts/ndvi_evaluation")
    p.set_defaults(func=cmd_ndvi)

    p = sub.add_parser("rank-overlap", help="Rank samples by cloud overlap across all selected inputs")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    p.add_argument("--n-input-times", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    p.add_argument("--output-dir", default="results_sen12mscrts/sample_ranking_overlap")
    p.set_defaults(func=cmd_rank_overlap)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
