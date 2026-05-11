from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_zero_one(x: torch.Tensor | np.ndarray, normalized_minus1_1: bool = True) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = x.astype(np.float32)
    if normalized_minus1_1:
        x = (x + 1.0) / 2.0
    return np.clip(x, 0.0, 1.0)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 3:
        scores = []
        for channel in range(pred.shape[0]):
            scores.append(structural_similarity(target[channel], pred[channel], data_range=1.0))
        return float(np.mean(scores))
    return float(structural_similarity(target, pred, data_range=1.0))
