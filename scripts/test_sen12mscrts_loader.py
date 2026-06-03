from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.dataset.sen12mscrts_loader import SEN12MSCRTSDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test SEN12MS-CR-TS loader")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--region", default="all", choices=["all", "africa", "america", "asiaEast", "asiaWest", "europa"])
    parser.add_argument("--n-input-times", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--cloud-detector", default="s2cloudless", choices=["s2cloudless", "heuristic"])
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = SEN12MSCRTSDataset(
        root_dir=args.root_dir,
        split=args.split,
        region=args.region,
        n_input_times=args.n_input_times,
        max_samples=args.max_samples,
        cloud_detector=args.cloud_detector,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    print(f"Dataset samples indexed: {len(dataset)}")
    print("masked_input:", tuple(batch["masked_input"].shape))
    print("target:", tuple(batch["target"].shape))
    print("mask:", tuple(batch["mask"].shape))
    print("roi_name:", batch["roi_name"])
    print("input_times:", batch["input_times"])
    print("target_time:", batch["target_time"])


if __name__ == "__main__":
    main()
