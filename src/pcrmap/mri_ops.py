from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2 for real/imag, got {x.shape}")
    return torch.view_as_complex(x.contiguous())


def from_complex(x: torch.Tensor) -> torch.Tensor:
    if not x.is_complex():
        raise ValueError("Expected complex tensor")
    return torch.view_as_real(x)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    X = torch.fft.fftshift(X, dim=(-2, -1))
    return X


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    X = torch.fft.ifftshift(X, dim=(-2, -1))
    x = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def center_crop(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    h, w = shape
    H, W = x.shape[-2], x.shape[-1]
    if h > H or w > W:
        raise ValueError(f"Requested crop {shape} larger than input {(H, W)}")
    top = (H - h) // 2
    left = (W - w) // 2
    return x[..., top:top + h, left:left + w]


def rss(coil_imgs: torch.Tensor, dim: int = -3, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.abs(coil_imgs) ** 2, dim=dim) + eps)


def make_vd_cartesian_mask(
    shape: Tuple[int, int],
    accel: int,
    center_fraction: float,
    seed: int = 0,
) -> torch.Tensor:
    H, W = shape
    rng = torch.Generator()
    rng.manual_seed(int(seed))

    num_low = int(round(W * center_fraction))
    num_low = max(2, min(W, num_low))
                          
    pad = (W - num_low) // 2
    acs = torch.zeros(W, dtype=torch.bool)
    acs[pad:pad + num_low] = True

                                     
    target = max(num_low, int(round(W / accel)))
    remain = max(0, target - num_low)

    probs = torch.linspace(0.0, 1.0, W)
    probs = torch.minimum(probs, 1.0 - probs)                         
    probs = probs / probs.sum()

    candidates = (~acs).nonzero(as_tuple=False).squeeze(-1)
    cand_probs = probs[candidates]
    cand_probs = cand_probs / cand_probs.sum()

    if remain > 0:
        idx = torch.multinomial(cand_probs, num_samples=remain, replacement=False, generator=rng)
        chosen = candidates[idx]
    else:
        chosen = torch.tensor([], dtype=torch.long)

    m = acs.clone()
    m[chosen] = True

    mask_1d = m[None, :].repeat(H, 1)
    return mask_1d


def estimate_sens_maps_from_acs(
    kspace_us: torch.Tensor,
    acs_fraction: float = 0.08,
    eps: float = 1e-8,
) -> torch.Tensor:
    C, H, W = kspace_us.shape
    num_low = int(round(W * acs_fraction))
    num_low = max(2, min(W, num_low))
    pad = (W - num_low) // 2

    k_acs = torch.zeros_like(kspace_us)
    k_acs[..., pad:pad + num_low] = kspace_us[..., pad:pad + num_low]

    coil_lr = ifft2c(k_acs)                             
    denom = rss(coil_lr, dim=0, eps=eps)               
    sens = coil_lr / (denom[None, ...] + eps)
    return sens


def sense_combine(
    coil_imgs: torch.Tensor,                     
    sens: torch.Tensor,                          
    eps: float = 1e-8,
) -> torch.Tensor:
    num = torch.sum(torch.conj(sens) * coil_imgs, dim=0)
    den = torch.sum(torch.abs(sens) ** 2, dim=0) + eps
    return num / den


def forward_op(x: torch.Tensor, sens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    coil_imgs = sens * x[None, ...]
    k = fft2c(coil_imgs)
    return k * mask[None, ...]


def adjoint_op(kspace: torch.Tensor, sens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    coil_imgs = ifft2c(kspace * mask[None, ...])
    x = torch.sum(torch.conj(sens) * coil_imgs, dim=0)
    return x


def smooth2d(x: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    if ksize <= 1:
        return x
    pad = ksize // 2
    x4 = x[None, None, ...]
    y = F.avg_pool2d(F.pad(x4, (pad, pad, pad, pad), mode="reflect"), kernel_size=ksize, stride=1)
    return y[0, 0]
