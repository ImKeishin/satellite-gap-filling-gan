from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.gan import build_models
from src.utils.metrics import to_zero_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generator on one .npy tensor")
    parser.add_argument("--input", required=True, help="Path to .npy file with shape (C,H,W) or (H,W)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="results/inference_output.npy")
    parser.add_argument("--base-channels", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arr = np.load(args.input).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("Input must have shape (C,H,W) or (H,W)")
    if arr.max() > 1.0:
        arr = arr / (10000.0 if arr.max() > 255.0 else 255.0)
    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr * 2.0 - 1.0).unsqueeze(0).float().to(device)

    generator, _ = build_models(in_channels=arr.shape[0], out_channels=arr.shape[0], base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    generator.eval()

    with torch.no_grad():
        pred = generator(tensor)[0]
    output = to_zero_one(pred, normalized_minus1_1=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, output)
    print(f"Saved reconstructed tensor to: {output_path}")


if __name__ == "__main__":
    main()
