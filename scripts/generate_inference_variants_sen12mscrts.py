from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.models.gan import build_models
from src.utils.metrics import mae, psnr, ssim, to_zero_one
from src.utils.visualization import save_temporal_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple easy, medium and hard inference examples for SEN12MS-CR-TS."
    )

    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="africa")
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--ranking-samples", type=int, default=256)
    parser.add_argument("--variants-per-difficulty", type=int, default=8)

    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument(
        "--input-selection",
        default="random_all",
        choices=["top_cloudy", "random_all", "least_cloudy"],
    )
    parser.add_argument("--input-random-seed", type=int, default=42)
    parser.add_argument("--min-input-cloud-coverage", type=float, default=0.05)
    parser.add_argument("--max-input-cloud-coverage", type=float, default=0.95)

    parser.add_argument("--output-root", default="results_sen12mscrts")
    parser.add_argument("--prefix", default="final_inference_random_all_r3")

    return parser.parse_args()


def to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def unique_quantile_positions(total: int, start_q: float, end_q: float, count: int) -> list[int]:
    start = max(0, int(round(start_q * (total - 1))))
    end = min(total - 1, int(round(end_q * (total - 1))))

    if end < start:
        end = start

    positions = np.linspace(start, end, count).round().astype(int).tolist()

    unique = []
    for pos in positions:
        if pos not in unique:
            unique.append(pos)

    candidate = start
    while len(unique) < count and candidate <= end:
        if candidate not in unique:
            unique.append(candidate)
        candidate += 1

    return unique[:count]


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        max_samples=args.ranking_samples,
        cloud_detector=args.cloud_detector,
        input_selection=args.input_selection,
        input_sampling_repeats=1,
        random_seed=args.input_random_seed,
        min_input_cloud_coverage=args.min_input_cloud_coverage,
        max_input_cloud_coverage=args.max_input_cloud_coverage,
    )

    generator, _ = build_models(args.n_input_times * 13, 13, args.base_channels)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()

    print(f"Indexez {len(dataset)} eșantioane după gradul de dificultate...")

    ranked_rows = []

    for index in range(len(dataset)):
        item = dataset[index]

        input_times = item["input_times"].detach().cpu().numpy().astype(int).tolist()
        cloud_coverages = item["cloud_coverages"].detach().cpu().numpy()

        input_coverages = [float(cloud_coverages[t]) for t in input_times]
        target_time = to_int(item["target_time"])
        target_coverage = float(cloud_coverages[target_time])
        union_coverage = float(item["mask"].float().mean().item())

        ranked_rows.append(
            {
                "sample_index": index,
                "roi_name": item["roi_name"],
                "patch_index": int(item["patch_index"]),
                "input_times": ",".join(str(t) for t in input_times),
                "target_time": target_time,
                "input_coverages": ",".join(f"{x:.6f}" for x in input_coverages),
                "average_input_coverage": sum(input_coverages) / len(input_coverages),
                "target_coverage": target_coverage,
                "union_coverage": union_coverage,
                "difficulty_score": union_coverage,
            }
        )

        print(f"\rProcesat {index + 1}/{len(dataset)}", end="", flush=True)

    print()

    ranked_rows.sort(key=lambda row: float(row["difficulty_score"]))

    difficulty_ranges = {
        "easy": (0.05, 0.25),
        "medium": (0.40, 0.60),
        "hard": (0.75, 0.95),
    }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_rows = []

    for difficulty, (start_q, end_q) in difficulty_ranges.items():
        output_dir = output_root / f"{args.prefix}_{difficulty}"
        output_dir.mkdir(parents=True, exist_ok=True)

        positions = unique_quantile_positions(
            total=len(ranked_rows),
            start_q=start_q,
            end_q=end_q,
            count=args.variants_per_difficulty,
        )

        print(f"\nGenerez variante pentru {difficulty}: {output_dir}")

        for variant_id, position in enumerate(positions, start=1):
            row = ranked_rows[position]
            sample_index = int(row["sample_index"])

            item = dataset[sample_index]

            condition = item["masked_input"].unsqueeze(0).to(device)
            target = item["target"]

            prediction = generator(condition)[0]

            cond_np = to_zero_one(item["masked_input"], True)
            pred_np = to_zero_one(prediction, True)
            target_np = to_zero_one(target, True)

            comparison_name = f"comparison_{variant_id:02d}_sample_{sample_index:04d}.png"
            prediction_name = f"prediction_13bands_{variant_id:02d}_sample_{sample_index:04d}.npy"
            metadata_name = f"metadata_{variant_id:02d}_sample_{sample_index:04d}.txt"

            save_temporal_result(
                cond_np,
                pred_np,
                target_np,
                output_dir / comparison_name,
                args.n_input_times,
            )

            np.save(output_dir / prediction_name, pred_np)

            sample_mae = mae(pred_np, target_np)
            sample_psnr = psnr(pred_np, target_np)
            sample_ssim = ssim(pred_np, target_np)

            metadata = {
                **row,
                "difficulty": difficulty,
                "variant_id": variant_id,
                "rank_position": position,
                "comparison_file": comparison_name,
                "prediction_file": prediction_name,
                "mae": sample_mae,
                "psnr": sample_psnr,
                "ssim": sample_ssim,
            }

            with open(output_dir / metadata_name, "w", encoding="utf-8") as file:
                for key, value in metadata.items():
                    file.write(f"{key}: {value}\n")

            selected_rows.append(metadata)

            print(
                f"  {difficulty:6s} #{variant_id:02d} | "
                f"sample={sample_index:04d} | "
                f"difficulty={float(row['difficulty_score']):.4f} | "
                f"MAE={sample_mae:.6f} | "
                f"PSNR={sample_psnr:.3f} | "
                f"SSIM={sample_ssim:.4f}"
            )

    ranking_csv = output_root / f"{args.prefix}_all_ranked_samples.csv"
    with open(ranking_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ranked_rows[0].keys())
        writer.writeheader()
        writer.writerows(ranked_rows)

    selected_csv = output_root / f"{args.prefix}_selected_variants.csv"
    with open(selected_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=selected_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"\nGata.")
    print(f"Ranking complet: {ranking_csv}")
    print(f"Variante selectate: {selected_csv}")


if __name__ == "__main__":
    main()