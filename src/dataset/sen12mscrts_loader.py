from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as exc:  # pragma: no cover
    rasterio = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

try:
    from s2cloudless import S2PixelCloudDetector
except ImportError:  # pragma: no cover
    S2PixelCloudDetector = None


S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# ROI split definitions follow the official SEN12MS-CR-TS toolbox.
ROI_GROUPS: dict[str, list[str]] = {
    "ROIs1158": ["106"],
    "ROIs1868": ["17", "36", "56", "73", "85", "100", "114", "119", "121", "126", "127", "139", "142", "143"],
    "ROIs1970": ["20", "21", "35", "40", "57", "65", "71", "82", "83", "91", "112", "116", "119", "128", "132", "133", "135", "139", "142", "144", "149"],
    "ROIs2017": ["8", "22", "25", "32", "49", "61", "63", "69", "75", "103", "108", "115", "116", "117", "130", "140", "146"],
}

REGION_ROIS: dict[str, list[str]] = {
    "africa": ["ROIs1970/21", "ROIs1970/35", "ROIs1970/40", "ROIs2017/8", "ROIs2017/22", "ROIs2017/32", "ROIs2017/61", "ROIs2017/75", "ROIs2017/140"],
    "america": ["ROIs1158/106", "ROIs1868/36", "ROIs1868/85", "ROIs1970/65", "ROIs1970/82", "ROIs1970/132", "ROIs1970/142", "ROIs2017/49", "ROIs2017/116"],
    "asiaEast": ["ROIs1868/73", "ROIs1868/114", "ROIs1868/119", "ROIs1868/126", "ROIs1868/143", "ROIs1970/116", "ROIs1970/135", "ROIs1970/139", "ROIs2017/25", "ROIs2017/117"],
    "asiaWest": ["ROIs1868/100", "ROIs1868/127", "ROIs1970/57", "ROIs1970/83", "ROIs1970/112", "ROIs2017/69", "ROIs2017/115", "ROIs2017/130"],
    "europa": ["ROIs1868/17", "ROIs1868/56", "ROIs1868/121", "ROIs1868/139", "ROIs1868/142", "ROIs1970/20", "ROIs1970/71", "ROIs1970/91", "ROIs1970/119", "ROIs1970/128", "ROIs1970/133", "ROIs1970/144", "ROIs1970/149", "ROIs2017/63", "ROIs2017/103", "ROIs2017/108", "ROIs2017/146"],
}

ALL_ROIS = [f"{group}/{roi}" for group, rois in ROI_GROUPS.items() for roi in rois]
TEST_ROIS = [
    "ROIs1868/119", "ROIs1970/139", "ROIs2017/108", "ROIs2017/63", "ROIs1158/106",
    "ROIs1868/73", "ROIs2017/32", "ROIs1868/100", "ROIs1970/132", "ROIs2017/103",
    "ROIs1868/142", "ROIs1970/20", "ROIs2017/140",
]
VAL_ROIS = ["ROIs2017/22", "ROIs1970/65", "ROIs2017/117", "ROIs1868/127", "ROIs1868/17"]
TRAIN_ROIS = [roi for roi in ALL_ROIS if roi not in TEST_ROIS and roi not in VAL_ROIS]
SPLITS = {"train": TRAIN_ROIS, "val": VAL_ROIS, "test": TEST_ROIS, "all": ALL_ROIS}


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


@dataclass(frozen=True)
class Sample:
    roi: str
    patch_index: int
    s2_paths: tuple[Path, ...]


class SEN12MSCRTSDataset(Dataset):
    """Sentinel-2-only temporal loader for SEN12MS-CR-TS.

    Expected extracted structure::

        root/ROIs1970/21/S2/0/*.tif
        root/ROIs1970/21/S2/1/*.tif
        ...
        root/ROIs1970/21/S2/29/*.tif

    One item returns:
        masked_input: (n_input_times * 13, H, W)
        target:       (13, H, W)
        mask:         (1, H, W), union of cloud masks from selected inputs

    The least-cloudy time point becomes the target and cloudier observations
    become temporal conditions, matching the cloudy -> cloud-free use case.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "val", "test", "all"] = "train",
        region: Literal["all", "africa", "america", "asiaEast", "asiaWest", "europa"] = "all",
        roi_filter: Iterable[str] | None = None,
        n_input_times: int = 3,
        time_count: int = 30,
        normalize: Literal["minus1_1", "zero_one", "none"] = "minus1_1",
        sentinel_scale: float = 10000.0,
        cloud_detector: Literal["s2cloudless", "heuristic"] = "s2cloudless",
        cloud_threshold: float = 0.4,
        min_input_cloud_coverage: float = 0.05,
        max_input_cloud_coverage: float = 1.0,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        if rasterio is None:
            raise ImportError("Install rasterio first: pip install rasterio") from _RASTERIO_IMPORT_ERROR
        if n_input_times < 1 or n_input_times >= time_count:
            raise ValueError("n_input_times must be >= 1 and smaller than time_count")
        if cloud_detector == "s2cloudless" and S2PixelCloudDetector is None:
            raise ImportError("Install s2cloudless first: pip install s2cloudless, or run with --cloud-detector heuristic")

        self.root_dir = Path(root_dir)
        self.n_input_times = n_input_times
        self.time_count = time_count
        self.normalize = normalize
        self.sentinel_scale = sentinel_scale
        self.cloud_detector_name = cloud_detector
        self.min_input_cloud_coverage = min_input_cloud_coverage
        self.max_input_cloud_coverage = max_input_cloud_coverage
        self.cloud_detector = None
        if cloud_detector == "s2cloudless":
            self.cloud_detector = S2PixelCloudDetector(
                threshold=cloud_threshold,
                all_bands=True,
                average_over=4,
                dilation_size=2,
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        selected = list(SPLITS[split])
        if region != "all":
            allowed = set(REGION_ROIS[region])
            selected = [roi for roi in selected if roi in allowed]
        if roi_filter is not None:
            requested = set(roi_filter)
            selected = [roi for roi in selected if roi in requested]

        self.samples = self._build_index(selected)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(
                "No samples found. Check the extracted folder structure. Expected e.g. "
                "root/ROIs1970/21/S2/0/*.tif through root/ROIs1970/21/S2/29/*.tif."
            )

    def _build_index(self, selected_rois: list[str]) -> list[Sample]:
        samples: list[Sample] = []
        for roi in selected_rois:
            group, roi_id = roi.split("/")
            s2_root = self.root_dir / group / roi_id / "S2"
            if not s2_root.exists():
                continue
            time_files: list[list[Path]] = []
            for t in range(self.time_count):
                folder = s2_root / str(t)
                if not folder.exists():
                    time_files = []
                    break
                files = sorted(folder.glob("*.tif"), key=_natural_key)
                if not files:
                    time_files = []
                    break
                time_files.append(files)
            if not time_files:
                continue
            patch_count = min(len(paths) for paths in time_files)
            for patch_idx in range(patch_count):
                samples.append(Sample(roi, patch_idx, tuple(time_files[t][patch_idx] for t in range(self.time_count))))
        return samples

    @staticmethod
    def _read_s2(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        if arr.ndim != 3 or arr.shape[0] < 13:
            raise ValueError(f"Expected 13-band GeoTIFF, got shape {arr.shape}: {path}")
        return arr[:13]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if self.normalize == "none":
            return arr
        arr = np.clip(arr, 0.0, self.sentinel_scale) / self.sentinel_scale
        if self.normalize == "minus1_1":
            arr = arr * 2.0 - 1.0
        return arr.astype(np.float32)

    def _cloud_mask(self, raw_s2: np.ndarray) -> np.ndarray:
        scaled = np.clip(raw_s2, 0.0, self.sentinel_scale) / self.sentinel_scale
        if self.cloud_detector_name == "s2cloudless":
            image = np.moveaxis(scaled, 0, -1)[None, ...]
            return self.cloud_detector.get_cloud_masks(image)[0].astype(np.float32)

        # Lightweight fallback for testing and environments where LightGBM is
        # inconvenient to install. For thesis experiments prefer s2cloudless.
        blue, green, red = scaled[1], scaled[2], scaled[3]
        visible_mean = (blue + green + red) / 3.0
        whiteness = 1.0 - (
            np.abs(blue - visible_mean) + np.abs(green - visible_mean) + np.abs(red - visible_mean)
        ) / 3.0
        prob = np.clip(0.65 * visible_mean + 0.35 * whiteness, 0.0, 1.0)
        return (prob > 0.62).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        raw_sequence = [self._read_s2(path) for path in sample.s2_paths]
        masks = [self._cloud_mask(image) for image in raw_sequence]
        coverages = np.asarray([float(mask.mean()) for mask in masks], dtype=np.float32)

        target_time = int(np.argmin(coverages))
        ranked = [int(i) for i in np.argsort(-coverages) if int(i) != target_time]
        filtered = [
            i for i in ranked
            if self.min_input_cloud_coverage <= float(coverages[i]) <= self.max_input_cloud_coverage
        ]
        if len(filtered) < self.n_input_times:
            filtered = ranked
        input_times = filtered[: self.n_input_times]

        condition = np.concatenate([self._normalize(raw_sequence[t]) for t in input_times], axis=0)
        target = self._normalize(raw_sequence[target_time])
        union_mask = np.maximum.reduce([masks[t] for t in input_times])[None, ...].astype(np.float32)

        return {
            "masked_input": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(union_mask).float(),
            "roi_name": sample.roi,
            "patch_index": sample.patch_index,
            "input_times": torch.tensor(input_times, dtype=torch.long),
            "target_time": torch.tensor(target_time, dtype=torch.long),
            "cloud_coverages": torch.from_numpy(coverages),
        }
