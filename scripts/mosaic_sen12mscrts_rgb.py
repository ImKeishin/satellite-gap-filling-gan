from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def stretch_band_fixed(band: np.ndarray, scale_max: float) -> np.ndarray:
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=scale_max, neginf=0.0)
    band = np.clip(band, 0.0, scale_max) / scale_max
    return (band * 255.0 + 0.5).astype(np.uint8)


def read_tif_as_rgb(path: Path, scale_max: float) -> Image.Image:
    with rasterio.open(path) as src:
        if src.count >= 4:
            rgb = src.read([4, 3, 2]).astype(np.float32)
        elif src.count >= 3:
            rgb = src.read([3, 2, 1]).astype(np.float32)
        else:
            raise ValueError(f"Fisierul nu are suficiente benzi pentru RGB: {path}")

    out = np.stack([stretch_band_fixed(rgb[i], scale_max) for i in range(3)], axis=-1)
    return Image.fromarray(out, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--roi", required=True, help="Exemplu: ROIs2017/32")
    parser.add_argument("--time-index", type=int, required=True, help="Folder temporal: 0..29")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--scale-max", type=float, default=3000.0)
    parser.add_argument("--resize", type=float, default=1.0)

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    roi_parts = args.roi.replace("\\", "/").split("/")
    folder = root_dir.joinpath(*roi_parts) / "S2" / str(args.time_index)

    if not folder.exists():
        raise RuntimeError(f"Folderul nu exista: {folder}")

    files = sorted(folder.glob("*.tif"), key=natural_key)
    if not files:
        raise RuntimeError(f"Nu am gasit patch-uri .tif in: {folder}")

    first = read_tif_as_rgb(files[0], args.scale_max)
    patch_w, patch_h = first.size

    cols = args.cols
    if cols is None:
        root = int(math.sqrt(len(files)))
        cols = root if root * root == len(files) else math.ceil(math.sqrt(len(files)))

    rows = math.ceil(len(files) / cols)

    mosaic = Image.new("RGB", (cols * patch_w, rows * patch_h), (0, 0, 0))

    for idx, path in enumerate(tqdm(files, desc="Construire mosaic")):
        img = first if idx == 0 else read_tif_as_rgb(path, args.scale_max)

        x = (idx % cols) * patch_w
        y = (idx // cols) * patch_h
        mosaic.paste(img, (x, y))

    if args.resize != 1.0:
        new_size = (int(mosaic.width * args.resize), int(mosaic.height * args.resize))
        mosaic = mosaic.resize(new_size, Image.Resampling.BILINEAR)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)

    print(f"Patch-uri combinate: {len(files)}")
    print(f"Grid: {cols} x {rows}")
    print(f"Dimensiune imagine: {mosaic.width} x {mosaic.height}")
    print(f"Salvat: {out_path}")


if __name__ == "__main__":
    main()