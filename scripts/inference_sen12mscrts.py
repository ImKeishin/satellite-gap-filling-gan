from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset
from src.models.gan import build_models
from src.utils.metrics import to_zero_one
from src.utils.visualization import save_temporal_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on one indexed SEN12MS-CR-TS sample")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints_sen12mscrts/best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="all", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    parser.add_argument(
        "--input-selection",
        default="random_all",
        choices=["top_cloudy", "random_all", "least_cloudy"],
    )
    parser.add_argument("--input-random-seed", type=int, default=42)
    parser.add_argument("--min-input-cloud-coverage", type=float, default=0.05)
    parser.add_argument("--max-input-cloud-coverage", type=float, default=0.95)
    parser.add_argument("--output-dir", default="results_sen12mscrts/inference")
    return parser.parse_args()

def _to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        cloud_detector=args.cloud_detector,
        input_selection=args.input_selection,
        input_sampling_repeats=1,
        random_seed=args.input_random_seed,
        min_input_cloud_coverage=args.min_input_cloud_coverage,
        max_input_cloud_coverage=args.max_input_cloud_coverage,
   )
    item = dataset[args.sample_index]
    condition = item["masked_input"].unsqueeze(0).to(device)
    target = item["target"]
    generator, _ = build_models(args.n_input_times * 13, 13, args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()
    prediction = generator(condition)[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_np = to_zero_one(prediction, True)
    cond_np = to_zero_one(item["masked_input"], True)
    target_np = to_zero_one(target, True)
    np.save(output_dir / "prediction_13bands.npy", pred_np)
    save_temporal_result(cond_np, pred_np, target_np, output_dir / "comparison.png", args.n_input_times)
    input_times = item["input_times"].detach().cpu().numpy().astype(int).tolist()
    target_time = _to_int(item["target_time"])
    coverages = item["cloud_coverages"].detach().cpu().numpy()

    input_coverages = [float(coverages[t]) for t in input_times]
    target_coverage = float(coverages[target_time])
    union_coverage = float(item["mask"].mean().item())

    with open(output_dir / "metadata.txt", "w", encoding="utf-8") as file:
        file.write(f"sample_index: {args.sample_index}\n")
        file.write(f"roi_name: {item['roi_name']}\n")
        file.write(f"patch_index: {item['patch_index']}\n")
        file.write(f"input_selection: {args.input_selection}\n")
        file.write(f"input_random_seed: {args.input_random_seed}\n")
        file.write(f"input_times: {input_times}\n")
        file.write(f"target_time: {target_time}\n")
        file.write(f"input_coverages: {input_coverages}\n")
        file.write(f"average_input_coverage: {sum(input_coverages) / len(input_coverages):.6f}\n")
        file.write(f"target_coverage: {target_coverage:.6f}\n")
        file.write(f"union_coverage: {union_coverage:.6f}\n")
    print(f"Saved 13-band reconstruction and RGB comparison in: {output_dir}")


if __name__ == "__main__":
    main()
