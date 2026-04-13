from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleIndex:
    roi_name: str
    t: int


class SentinelNPYDataset(Dataset):
    """
    Dataset for Sentinel-2 spectral bands stored as separate .npy files.

    Expected structure:
        root_dir/
            B01_SR/
                roi1.npy
                roi2.npy
            B02_SR/
                roi1.npy
                roi2.npy
            B03_SR/
                roi1.npy
                roi2.npy
            ...

    Each .npy file is expected to have shape:
        (H, W, T)

    where:
        H, W = spatial dimensions
        T    = temporal dimension

    For one sample, the dataset returns:
        masked_input: (B, H, W)
        target:       (B, H, W)
        mask:         (1, H, W)
        meta:         dict with roi_name and t
    """

    def __init__(
        self,
        root_dir: str,
        bands: List[str],
        rois: Optional[List[str]] = None,
        normalize: str = "minus1_1",
        create_synthetic_mask: bool = True,
        mask_ratio_range: Tuple[float, float] = (0.10, 0.30),
        seed: int = 42,
        preload: bool = False,
    ) -> None:
        """
        Args:
            root_dir: Path to folder containing Bxx_SR folders.
            bands: Example: ["B01_SR", "B02_SR", "B03_SR", "B04_SR"]
            rois: Optional list like ["roi1", "roi2"]. If None, intersection is auto-detected.
            normalize:
                - "none"      -> leave as is
                - "zero_one"  -> divide by 255 if max > 1
                - "minus1_1"  -> scale to [-1, 1]
            create_synthetic_mask:
                If True, creates random spatial masks for training inpainting/gap filling.
            mask_ratio_range:
                Fraction of pixels to hide, between low/high.
            seed:
                Random seed for reproducibility.
            preload:
                If True, loads all arrays in RAM at init.
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.bands = bands
        self.normalize = normalize
        self.create_synthetic_mask = create_synthetic_mask
        self.mask_ratio_range = mask_ratio_range
        self.preload = preload

        self.rng = np.random.default_rng(seed)

        self._validate_root()
        self.band_to_roi_paths = self._collect_band_roi_paths()
        self.roi_names = self._resolve_rois(rois)
        self.sample_index = self._build_sample_index()

        self.data_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if self.preload:
            self._preload_arrays()

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

    def _collect_band_roi_paths(self) -> Dict[str, Dict[str, Path]]:
        band_to_roi_paths: Dict[str, Dict[str, Path]] = {}

        for band in self.bands:
            band_dir = self.root_dir / band
            if not band_dir.exists():
                raise FileNotFoundError(f"Band folder not found: {band_dir}")

            roi_paths: Dict[str, Path] = {}
            for npy_path in sorted(band_dir.glob("*.npy")):
                roi_name = npy_path.stem
                roi_paths[roi_name] = npy_path

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

    def _safe_load_npy(self, path: Path) -> np.ndarray:
        try:
            arr = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, T), got {arr.shape} in {path}")

        return arr

    def _preload_arrays(self) -> None:
        for band in self.bands:
            self.data_cache[band] = {}
            for roi_name in self.roi_names:
                path = self.band_to_roi_paths[band][roi_name]
                arr = self._safe_load_npy(path)
                self.data_cache[band][roi_name] = arr

    def _get_array(self, band: str, roi_name: str) -> np.ndarray:
        if self.preload:
            return self.data_cache[band][roi_name]
        return self._safe_load_npy(self.band_to_roi_paths[band][roi_name])

    def _build_sample_index(self) -> List[SampleIndex]:
        sample_index: List[SampleIndex] = []

        for roi_name in self.roi_names:
            reference_shapes = []
            for band in self.bands:
                arr = self._get_array(band, roi_name)
                reference_shapes.append(arr.shape)

            first_shape = reference_shapes[0]
            for shape in reference_shapes[1:]:
                if shape != first_shape:
                    raise ValueError(
                        f"Shape mismatch for ROI '{roi_name}' across bands: {reference_shapes}"
                    )

            _, _, t_count = first_shape
            for t in range(t_count):
                sample_index.append(SampleIndex(roi_name=roi_name, t=t))

        return sample_index

    def __len__(self) -> int:
        return len(self.sample_index)

    def _normalize_array(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self.normalize == "none":
            return x

        if self.normalize == "zero_one":
            if x.max() > 1.0:
                x = x / 255.0
            return x

        if self.normalize == "minus1_1":
            if x.max() > 1.0:
                x = x / 255.0
            x = (x * 2.0) - 1.0
            return x

        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _create_random_mask(self, height: int, width: int) -> np.ndarray:
        """
        Creates a simple random rectangular mask.
        Output shape: (1, H, W), values in {0,1}
        1 = missing region
        """
        low, high = self.mask_ratio_range
        target_ratio = float(self.rng.uniform(low, high))
        target_area = int(target_ratio * height * width)

        mask = np.zeros((height, width), dtype=np.float32)
        filled = 0

        while filled < target_area:
            rect_h = int(self.rng.integers(max(8, height // 16), max(16, height // 4)))
            rect_w = int(self.rng.integers(max(8, width // 16), max(16, width // 4)))

            y0 = int(self.rng.integers(0, max(1, height - rect_h)))
            x0 = int(self.rng.integers(0, max(1, width - rect_w)))

            mask[y0:y0 + rect_h, x0:x0 + rect_w] = 1.0
            filled = int(mask.sum())

        return mask[None, :, :]

    def _load_multiband_frame(self, roi_name: str, t: int) -> np.ndarray:
        """
        Returns one time step stacked over bands.
        Output shape: (B, H, W)
        """
        band_frames = []

        for band in self.bands:
            arr = self._get_array(band, roi_name)   # (H, W, T)
            frame = arr[:, :, t]                    # (H, W)
            band_frames.append(frame)

        stacked = np.stack(band_frames, axis=0)     # (B, H, W)
        stacked = self._normalize_array(stacked)
        return stacked

    def __getitem__(self, idx: int):
        sample = self.sample_index[idx]
        target = self._load_multiband_frame(sample.roi_name, sample.t)  # (B, H, W)

        _, h, w = target.shape

        if self.create_synthetic_mask:
            mask = self._create_random_mask(h, w)   # (1, H, W)
        else:
            mask = np.zeros((1, h, w), dtype=np.float32)

        masked_input = target * (1.0 - mask)

        return {
            "masked_input": torch.from_numpy(masked_input).float(),
            "target": torch.from_numpy(target).float(),
            "mask": torch.from_numpy(mask).float(),
            "roi_name": sample.roi_name,
            "t": sample.t,
        }