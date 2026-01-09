from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import torch

from .mri_ops import to_complex, center_crop, ifft2c


def _kspace_to_torch(k: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(k)
    t = to_complex(t)                       
    return t


def load_multicoil_h5_slice(
    h5_path: str,
    slice_idx: int,
) -> Dict[str, torch.Tensor]:
    with h5py.File(h5_path, "r") as f:
        k = f["kspace"][slice_idx]             
        kspace_full = _kspace_to_torch(k)           
        ref_rss = None
        if "reconstruction_rss" in f:
            ref = f["reconstruction_rss"][slice_idx]         
            ref_rss = torch.from_numpy(ref).float()
    return {"kspace_full": kspace_full, "ref_rss": ref_rss}


def build_zf_coil_images(
    kspace_us: torch.Tensor,
    ref_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    coil = ifft2c(kspace_us)                   
    if ref_shape is not None:
        coil = center_crop(coil, ref_shape)
    return coil
