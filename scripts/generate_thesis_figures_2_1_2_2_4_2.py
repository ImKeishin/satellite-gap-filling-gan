#!/usr/bin/env python3
"""
Generate data-based thesis figures 2.1, 2.2 and 4.2 for the Sentinel-2 temporal
cloud-removal thesis.

python scripts/generate_thesis_figures_2_1_2_2_4_2.py \
  --root-dir D:\\data\\sen12mscrts\\test \
  --split test \
  --region africa \
  --cloud-detector s2cloudless \
  --auto-pick \
  --output-dir thesis_figures_good

The script uses the repository's SEN12MSCRTSDataset, but forces the same temporal
selection policy described in the thesis: random_all, 3 input observations,
cloud coverage range [0.05, 0.95], input_sampling_repeats=3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.utils.metrics import to_zero_one

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
RGB_BANDS = (3, 2, 1)  # Sentinel-2 true color display: B4, B3, B2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export good thesis figures 2.1, 2.2 and 4.2")
    parser.add_argument("--root-dir", required=True, help="Folder that directly contains ROIs... directories")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument("--sample-index", type=int, default=0, help="Used when --auto-pick is not set")
    parser.add_argument("--auto-pick", action="store_true", help="Search for a visually useful sample")
    parser.add_argument("--max-search-samples", type=int, default=180, help="Limit for --auto-pick; s2cloudless can be slow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="thesis_figures_good")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--also-pdf", action="store_true", help="Save PDF copies next to PNG files")
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, max_samples: int | None = None) -> SEN12MSCRTSDataset:
    return SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        min_input_cloud_coverage=0.05,
        max_input_cloud_coverage=0.95,
        input_selection="random_all",
        input_sampling_repeats=3,
        random_seed=args.seed,
        max_samples=max_samples,
    )


def split_condition(condition: np.ndarray, n_input_times: int) -> list[np.ndarray]:
    return [condition[i * 13 : (i + 1) * 13] for i in range(n_input_times)]


def rgb_percentile_stretch(image_13: np.ndarray, p_low: float = 2.0, p_high: float = 98.0, gamma: float = 0.9) -> np.ndarray:
    """Convert a normalized Sentinel-2 tensor (13,H,W) in [0,1] to a good-looking RGB image."""
    rgb = np.stack([image_13[idx] for idx in RGB_BANDS], axis=-1).astype(np.float32)

    # One stretch per displayed image gives robust figures even when scenes differ a lot in brightness.
    lo = np.nanpercentile(rgb, p_low)
    hi = np.nanpercentile(rgb, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0

    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    rgb = np.power(rgb, gamma)
    return rgb


def read_selected_masks(dataset: SEN12MSCRTSDataset, sample_index: int, input_times: list[int]) -> list[np.ndarray]:
    sample = dataset.samples[sample_index]
    selected_masks: list[np.ndarray] = []
    for t in input_times:
        raw_s2 = dataset._read_s2(sample.s2_paths[t])
        selected_masks.append(dataset._cloud_mask(raw_s2).astype(np.float32))
    return selected_masks


def load_payload(dataset: SEN12MSCRTSDataset, sample_index: int) -> dict[str, Any]:
    item = dataset[sample_index]
    condition = to_zero_one(item["masked_input"], True)
    target = to_zero_one(item["target"], True)
    input_times = [int(x) for x in item["input_times"].tolist()]
    target_time = int(item["target_time"])
    coverages = item["cloud_coverages"].detach().cpu().numpy().astype(float)
    selected_masks = read_selected_masks(dataset, sample_index, input_times)
    union_mask = np.maximum.reduce(selected_masks).astype(np.float32)
    inputs = split_condition(condition, len(input_times))
    selected_coverages = np.asarray([coverages[t] for t in input_times], dtype=float)
    cloudiest_input_pos = int(np.argmax(selected_coverages))
    return {
        "item": item,
        "inputs": inputs,
        "target": target,
        "input_times": input_times,
        "target_time": target_time,
        "coverages": coverages,
        "selected_coverages": selected_coverages,
        "selected_masks": selected_masks,
        "union_mask": union_mask,
        "cloudiest_input_pos": cloudiest_input_pos,
        "sample_index": sample_index,
    }


def score_payload(payload: dict[str, Any]) -> float:
    """Prefer samples with visible clouds, clean target, and a non-trivial aggregate mask."""
    union_cov = float(payload["union_mask"].mean())
    target_cov = float(payload["coverages"][payload["target_time"]])
    mean_input_cov = float(np.mean(payload["selected_coverages"]))
    max_input_cov = float(np.max(payload["selected_coverages"]))

    # Ideal for figures: target almost clear, inputs visibly cloudy, union mask neither empty nor fully white.
    score = 0.0
    score += abs(union_cov - 0.45) * 2.0
    score += abs(mean_input_cov - 0.35) * 1.0
    score += max(0.0, target_cov - 0.12) * 4.0
    score += max(0.0, 0.18 - max_input_cov) * 2.0
    score += max(0.0, union_cov - 0.80) * 2.0
    score += max(0.0, 0.15 - union_cov) * 2.0
    return score


def pick_sample(dataset: SEN12MSCRTSDataset, max_search_samples: int) -> tuple[int, dict[str, Any]]:
    best_index = 0
    best_payload = load_payload(dataset, 0)
    best_score = score_payload(best_payload)
    limit = min(max_search_samples, len(dataset))

    for index in range(1, limit):
        payload = load_payload(dataset, index)
        score = score_payload(payload)
        if score < best_score:
            best_index = index
            best_payload = payload
            best_score = score
        print(f"\rSearched {index + 1}/{limit}; best sample={best_index}, score={best_score:.4f}", end="", flush=True)
    print()
    return best_index, best_payload


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int, also_pdf: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    if also_pdf:
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {png_path}")


def export_fig_2_1(payload: dict[str, Any], output_dir: Path, dpi: int, also_pdf: bool) -> None:
    inputs = payload["inputs"]
    target = payload["target"]
    input_times = payload["input_times"]
    selected_coverages = payload["selected_coverages"]
    target_time = payload["target_time"]
    target_cov = float(payload["coverages"][target_time])

    fig, axes = plt.subplots(1, len(inputs) + 1, figsize=(4.2 * (len(inputs) + 1), 4.3))
    for i, ax in enumerate(axes[:-1]):
        ax.imshow(rgb_percentile_stretch(inputs[i]))
        ax.set_title(f"Intrare {i + 1}\nt={input_times[i]}, nori={selected_coverages[i] * 100:.1f}%", fontsize=11)
        ax.axis("off")

    axes[-1].imshow(rgb_percentile_stretch(target))
    axes[-1].set_title(f"Țintă\nt={target_time}, nori={target_cov * 100:.1f}%", fontsize=11)
    axes[-1].axis("off")
    fig.suptitle("Construirea unui eșantion temporal Sentinel-2", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_2_1_esantion_temporal", dpi, also_pdf)


def export_fig_2_2(payload: dict[str, Any], output_dir: Path, dpi: int, also_pdf: bool) -> None:
    pos = payload["cloudiest_input_pos"]
    cloudy_input = payload["inputs"][pos]
    target = payload["target"]
    union_mask = payload["union_mask"]
    input_time = payload["input_times"][pos]
    input_cov = float(payload["selected_coverages"][pos])
    union_cov = float(union_mask.mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    axes[0].imshow(rgb_percentile_stretch(cloudy_input))
    axes[0].set_title(f"Imagine afectată de nori\nt={input_time}, nori={input_cov * 100:.1f}%", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(union_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Mască agregată\nacoperire={union_cov * 100:.1f}%", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(rgb_percentile_stretch(target))
    axes[2].set_title("Imagine-țintă\nacoperire minimă de nori", fontsize=11)
    axes[2].axis("off")

    fig.suptitle("Intrare Sentinel-2, mască agregată și țintă", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_2_2_intrare_masca_tinta", dpi, also_pdf)


def export_fig_4_2(payload: dict[str, Any], output_dir: Path, dpi: int, also_pdf: bool) -> None:
    masks = payload["selected_masks"]
    union_mask = payload["union_mask"]
    input_times = payload["input_times"]

    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(4.0 * (len(masks) + 1), 4.2))
    for i, ax in enumerate(axes[:-1]):
        cov = float(masks[i].mean())
        ax.imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Mască intrare {i + 1}\nt={input_times[i]}, {cov * 100:.1f}%", fontsize=11)
        ax.axis("off")

    union_cov = float(union_mask.mean())
    axes[-1].imshow(union_mask, cmap="gray", vmin=0, vmax=1)
    axes[-1].set_title(f"Reuniunea măștilor\n{union_cov * 100:.1f}%", fontsize=11)
    axes[-1].axis("off")

    fig.suptitle("Construirea măștii agregate: M = M1 ∪ M2 ∪ M3", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_4_2_masca_agregata", dpi, also_pdf)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    search_limit = args.max_search_samples if args.auto_pick else None
    dataset = make_dataset(args, max_samples=search_limit)
    print(f"Indexed samples including repeats: {len(dataset)}")

    if args.auto_pick:
        sample_index, payload = pick_sample(dataset, args.max_search_samples)
    else:
        sample_index = args.sample_index
        payload = load_payload(dataset, sample_index)

    print("Selected figure sample:")
    print(f"  sample_index: {sample_index}")
    print(f"  ROI: {payload['item']['roi_name']}")
    print(f"  patch_index: {int(payload['item']['patch_index'])}")
    print(f"  input_times: {payload['input_times']}")
    print(f"  target_time: {payload['target_time']}")
    print(f"  input cloud coverages: {[round(float(x), 4) for x in payload['selected_coverages']]}")
    print(f"  union mask coverage: {float(payload['union_mask'].mean()):.4f}")
    print(f"  target cloud coverage: {float(payload['coverages'][payload['target_time']]):.4f}")

    export_fig_2_1(payload, output_dir, args.dpi, args.also_pdf)
    export_fig_2_2(payload, output_dir, args.dpi, args.also_pdf)
    export_fig_4_2(payload, output_dir, args.dpi, args.also_pdf)


if __name__ == "__main__":
    main()
