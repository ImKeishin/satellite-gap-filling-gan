from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleIndex:
    roi_name: str
    t: int


class SentinelNPYDataset(Dataset):
    """Dataset for Sentinel-2 spectral bands stored as separate .npy files.

    Expected structure:
        root_dir/B01_SR/roi1.npy
        root_dir/B02_SR/roi1.npy
        ...

    Each .npy file must have shape (H, W, T).
    Returned tensors:
        masked_input: (B, H, W)
        target:       (B, H, W)
        mask:         (1, H, W), where 1 means missing pixel
    """

    def __init__(
        self,
        root_dir: str,
        bands: List[str],
        rois: Optional[List[str]] = None,
        normalize: str = "minus1_1",
        sentinel_scale: float = 10000.0,
        create_synthetic_mask: bool = True,
        mask_ratio_range: Tuple[float, float] = (0.10, 0.30),
        seed: int = 42,
        preload: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.bands = bands
        self.normalize = normalize
        self.sentinel_scale = sentinel_scale
        self.create_synthetic_mask = create_synthetic_mask
        self.mask_ratio_range = mask_ratio_range
        self.preload = preload
        self.rng = np.random.default_rng(seed)
        self.data_cache: Dict[str, Dict[str, np.ndarray]] = {}

        self._validate_root()
        self.band_to_roi_paths = self._collect_band_roi_paths()
        self.roi_names = self._resolve_rois(rois)
        if self.preload:
            self._preload_arrays()
        self.sample_index = self._build_sample_index()

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

    def _collect_band_roi_paths(self) -> Dict[str, Dict[str, Path]]:
        band_to_roi_paths: Dict[str, Dict[str, Path]] = {}
        for band in self.bands:
            band_dir = self.root_dir / band
            if not band_dir.exists():
                raise FileNotFoundError(f"Band folder not found: {band_dir}")
            roi_paths = {p.stem: p for p in sorted(band_dir.glob("*.npy"))}
            if not roi_paths:
                raise ValueError(f"No .npy files found in {band_dir}")
            band_to_roi_paths[band] = roi_paths
        return band_to_roi_paths

    def _resolve_rois(self, rois: Optional[List[str]]) -> List[str]:
        available_sets = [set(v.keys()) for v in self.band_to_roi_paths.values()]
        common_rois = sorted(set.intersection(*available_sets))
        if not common_rois:
            raise ValueError("No common ROI files found across selected bands.")
        if rois is None:
            return common_rois
        missing = [r for r in rois if r not in common_rois]
        if missing:
            raise ValueError(f"Requested ROIs not found in all selected bands: {missing}")
        return rois

    @staticmethod
    def _safe_load_npy(path: Path) -> np.ndarray:
        try:
            arr = np.load(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, T), got {arr.shape} in {path}")
        return arr

    def _preload_arrays(self) -> None:
        for band in self.bands:
            self.data_cache[band] = {}
            for roi_name in self.roi_names:
                self.data_cache[band][roi_name] = self._safe_load_npy(self.band_to_roi_paths[band][roi_name])

    def _get_array(self, band: str, roi_name: str) -> np.ndarray:
        if self.preload:
            return self.data_cache[band][roi_name]
        return self._safe_load_npy(self.band_to_roi_paths[band][roi_name])

    def _build_sample_index(self) -> List[SampleIndex]:
        sample_index: List[SampleIndex] = []
        for roi_name in self.roi_names:
            shapes = [self._get_array(band, roi_name).shape for band in self.bands]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError(f"Shape mismatch for ROI '{roi_name}' across bands: {shapes}")
            _, _, t_count = shapes[0]
            for t in range(t_count):
                sample_index.append(SampleIndex(roi_name=roi_name, t=t))
        return sample_index

    def __len__(self) -> int:
        return len(self.sample_index)

    def _normalize_array(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if self.normalize == "none":
            return x
        if self.normalize in {"zero_one", "minus1_1"}:
            if np.nanmax(x) > 1.0:
                denominator = self.sentinel_scale if np.nanmax(x) > 255.0 else 255.0
                x = x / denominator
            x = np.clip(x, 0.0, 1.0)
            if self.normalize == "minus1_1":
                x = x * 2.0 - 1.0
            return x
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _create_random_mask(self, height: int, width: int) -> np.ndarray:
        low, high = self.mask_ratio_range
        target_area = int(float(self.rng.uniform(low, high)) * height * width)
        mask = np.zeros((height, width), dtype=np.float32)
        filled = 0
        min_h, max_h = max(4, height // 16), max(8, height // 4)
        min_w, max_w = max(4, width // 16), max(8, width // 4)

        while filled < target_area:
            rect_h = int(self.rng.integers(min_h, max_h + 1))
            rect_w = int(self.rng.integers(min_w, max_w + 1))
            rect_h = min(rect_h, height)
            rect_w = min(rect_w, width)
            y0 = int(self.rng.integers(0, max(1, height - rect_h + 1)))
            x0 = int(self.rng.integers(0, max(1, width - rect_w + 1)))
            mask[y0 : y0 + rect_h, x0 : x0 + rect_w] = 1.0
            filled = int(mask.sum())
        return mask[None, :, :]

    def _load_multiband_frame(self, roi_name: str, t: int) -> np.ndarray:
        frames = []
        for band in self.bands:
            arr = self._get_array(band, roi_name)
            frames.append(arr[:, :, t])
        stacked = np.stack(frames, axis=0)
        return self._normalize_array(stacked)

    def __getitem__(self, idx: int):
        sample = self.sample_index[idx]
        target = self._load_multiband_frame(sample.roi_name, sample.t)
        _, height, width = target.shape
        mask = self._create_random_mask(height, width) if self.create_synthetic_mask else np.zeros((1, height, width), dtype=np.float32)
        masked_input = target * (1.0 - mask)
        return {
            "masked_input": torch.from_numpy(masked_input).float(),
            "target": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(mask).float(),
            "roi_name": sample.roi_name,
            "t": sample.t,
        }
