from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def stretch_fixed(band: np.ndarray, scale_max: float) -> np.ndarray:
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=scale_max, neginf=0.0)
    band = np.clip(band, 0.0, scale_max) / scale_max
    return (band * 255.0 + 0.5).astype(np.uint8)


def read_rgb(path: Path, scale_max: float) -> Image.Image:
    with rasterio.open(path) as src:
        arr = src.read([4, 3, 2]).astype(np.float32)

    rgb = np.stack(
        [
            stretch_fixed(arr[0], scale_max),
            stretch_fixed(arr[1], scale_max),
            stretch_fixed(arr[2], scale_max),
        ],
        axis=-1,
    )

    return Image.fromarray(rgb, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--roi", required=True, help="Exemplu: ROIs2017/32")
    parser.add_argument("--time-index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale-max", type=float, default=3000.0)
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument("--debug-csv", default=None)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    roi_parts = args.roi.replace("\\", "/").split("/")
    folder = root_dir.joinpath(*roi_parts) / "S2" / str(args.time_index)

    files = sorted(folder.glob("*.tif"))

    if not files:
        raise RuntimeError(f"Nu am gasit fisiere .tif in {folder}")

    metadata = []

    for path in files:
        with rasterio.open(path) as src:
            transform = src.transform

            if transform.is_identity:
                raise RuntimeError(
                    f"{path} nu pare sa aiba georeferentiere valida. "
                    "In cazul asta nu putem reconstrui corect dupa coordonate."
                )

            metadata.append(
                {
                    "path": path,
                    "name": path.name,
                    "left": src.bounds.left,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                    "bottom": src.bounds.bottom,
                    "width": src.width,
                    "height": src.height,
                    "pixel_size_x": transform.a,
                    "pixel_size_y": transform.e,
                }
            )

    df = pd.DataFrame(metadata)

    patch_w = int(df["width"].iloc[0])
    patch_h = int(df["height"].iloc[0])

    xs = sorted(df["left"].unique())
    ys = sorted(df["top"].unique(), reverse=True)

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}

    df["col"] = df["left"].map(x_to_col)
    df["row"] = df["top"].map(y_to_row)

    cols = len(xs)
    rows = len(ys)

    mosaic = Image.new("RGB", (cols * patch_w, rows * patch_h), (0, 0, 0))

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Construire mosaic georef"):
        img = read_rgb(Path(row["path"]), args.scale_max)

        x = int(row["col"]) * patch_w
        y = int(row["row"]) * patch_h

        mosaic.paste(img, (x, y))

    if args.resize != 1.0:
        new_size = (
            int(mosaic.width * args.resize),
            int(mosaic.height * args.resize),
        )
        mosaic = mosaic.resize(new_size, Image.Resampling.BILINEAR)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)

    if args.debug_csv is not None:
        debug_path = Path(args.debug_csv)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["path"]).to_csv(debug_path, index=False)

    print(f"Folder: {folder}")
    print(f"Patch-uri: {len(files)}")
    print(f"Grid geografic: {cols} x {rows}")
    print(f"Dimensiune patch: {patch_w} x {patch_h}")
    print(f"Dimensiune mosaic: {mosaic.width} x {mosaic.height}")
    print(f"Salvat: {out_path}")

    expected_cells = rows * cols
    missing = expected_cells - len(files)

    if missing > 0:
        print(f"Atentie: exista {missing} celule lipsa in grid. Zonele respective raman negre.")


if __name__ == "__main__":
    main()