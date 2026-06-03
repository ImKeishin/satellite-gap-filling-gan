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
    parser.add_argument("--output-dir", default="results_sen12mscrts/inference")
    return parser.parse_args()


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
    print(f"Saved 13-band reconstruction and RGB comparison in: {output_dir}")


if __name__ == "__main__":
    main()
