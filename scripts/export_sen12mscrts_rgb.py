from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


def bytes_to_gb(value: int) -> float:
    return value / (1024 ** 3)


def stretch_band_fixed(band: np.ndarray, scale_max: float) -> np.ndarray:
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=scale_max, neginf=0.0)
    band = np.clip(band, 0.0, scale_max) / scale_max
    return (band * 255.0 + 0.5).astype(np.uint8)


def stretch_band_percentile(band: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(band, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    band = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return (band * 255.0 + 0.5).astype(np.uint8)


def read_tif_as_rgb(path: Path, stretch: str, scale_max: float, p_low: float, p_high: float) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count >= 4:
            rgb = src.read([4, 3, 2]).astype(np.float32)
        elif src.count >= 3:
            rgb = src.read([3, 2, 1]).astype(np.float32)
        else:
            raise ValueError(f"Fisierul nu are suficiente benzi pentru RGB: {path}")

    if stretch == "fixed":
        out = np.stack([stretch_band_fixed(rgb[i], scale_max) for i in range(3)], axis=-1)
    else:
        out = np.stack([stretch_band_percentile(rgb[i], p_low, p_high) for i in range(3)], axis=-1)

    return out


def convert_one(path: Path, root_dir: Path, out_dir: Path, args) -> tuple[int, int]:
    rel = path.relative_to(root_dir)
    suffix = ".jpg" if args.format == "jpg" else ".png"
    out_path = (out_dir / rel).with_suffix(suffix)

    if args.skip_existing and out_path.exists():
        return path.stat().st_size, out_path.stat().st_size

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = read_tif_as_rgb(
        path=path,
        stretch=args.stretch,
        scale_max=args.scale_max,
        p_low=args.p_low,
        p_high=args.p_high,
    )

    img = Image.fromarray(rgb, mode="RGB")

    if args.format == "jpg":
        img.save(out_path, quality=args.quality, optimize=True, subsampling=0)
    else:
        img.save(out_path, compress_level=args.png_compress_level)

    return path.stat().st_size, out_path.stat().st_size


def run_conversion(files: list[Path], root_dir: Path, out_dir: Path, args) -> tuple[int, int]:
    input_total = 0
    output_total = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(convert_one, path, root_dir, out_dir, args) for path in files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Conversie RGB"):
            in_size, out_size = future.result()
            input_total += in_size
            output_total += out_size

    return input_total, output_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--format", choices=["png", "jpg"], default="jpg")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--stretch", choices=["fixed", "percentile"], default="fixed")
    parser.add_argument("--scale-max", type=float, default=3000.0)
    parser.add_argument("--p-low", type=float, default=2.0)
    parser.add_argument("--p-high", type=float, default=98.0)
    parser.add_argument("--png-compress-level", type=int, default=6)

    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-sample", type=int, default=200)

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)

    files = sorted(root_dir.rglob("*.tif"))
    if not files:
        raise RuntimeError(f"Nu am gasit fisiere .tif in: {root_dir}")

    total_tif_size = sum(path.stat().st_size for path in files)

    print(f"Patch-uri gasite: {len(files)}")
    print(f"Dimensiune .tif gasita: {bytes_to_gb(total_tif_size):.2f} GB")

    if args.estimate_only:
        sample = files[: min(args.estimate_sample, len(files))]

        tmp_dir = Path(tempfile.mkdtemp(prefix="rgb_estimate_"))
        try:
            sample_in, sample_out = run_conversion(sample, root_dir, tmp_dir, args)
            ratio = sample_out / sample_in if sample_in else math.nan
            estimated_output = total_tif_size * ratio

            print(f"Sample convertit: {len(sample)} patch-uri")
            print(f"Raport estimat output/input: {ratio:.3f}")
            print(f"Output estimat: {bytes_to_gb(int(estimated_output)):.2f} GB")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return

    input_total, output_total = run_conversion(files, root_dir, out_dir, args)

    print(f"Input procesat: {bytes_to_gb(input_total):.2f} GB")
    print(f"Output generat: {bytes_to_gb(output_total):.2f} GB")
    print(f"Salvat in: {out_dir}")


if __name__ == "__main__":
    main()