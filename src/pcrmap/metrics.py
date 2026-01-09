from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None


def nrmse(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref) + eps
    return float(num / den)


def psnr(pred: np.ndarray, ref: np.ndarray, data_range: Optional[float] = None, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    mse = np.mean((pred - ref) ** 2)
    if data_range is None:
        data_range = float(ref.max() - ref.min())
        if data_range <= 0:
            data_range = float(ref.max() + eps)
    return float(20 * np.log10((data_range + eps) / (np.sqrt(mse) + eps)))


def ssim(pred: np.ndarray, ref: np.ndarray, data_range: Optional[float] = None) -> float:
    if ssim_fn is None:
        return float("nan")
    pred = pred.astype(np.float64)
    ref = ref.astype(np.float64)
    if data_range is None:
        data_range = float(ref.max() - ref.min())
        if data_range <= 0:
            data_range = float(ref.max())
    return float(ssim_fn(ref, pred, data_range=data_range))


def pearsonr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)
