from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .mri_ops import forward_op, adjoint_op, smooth2d, ifft2c, rss


@dataclass
class PCRConfig:
    eps: float = 1e-6
    smooth_ksize: int = 7
    clip_percentile: float = 99.5
    topk: float = 0.02
    core_box: Tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)                              


def compute_pcr_map(
    y: torch.Tensor,                                                   
    xhat: torch.Tensor,                                   
    sens: torch.Tensor,                      
    mask: torch.Tensor,                     
    cfg: PCRConfig = PCRConfig(),
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
                                                                                       
    dev = xhat.device
    y = y.to(dev)
    sens = sens.to(dev)
    mask = mask.to(dev)
    mask_f = mask.to(dtype=torch.float32)
                                                                                    

    k_pred = forward_op(xhat, sens, mask_f)
    resid_k = (k_pred - y) * mask_f[None, ...]
    resid_img = adjoint_op(resid_k, sens, mask_f)                 

                                                                                                 
                                                                                                    
    coil_zf = ifft2c(y)                               
    denom = rss(coil_zf, dim=0)                  
    denom = smooth2d(denom, ksize=cfg.smooth_ksize)
    pcr = torch.abs(resid_img) / (cfg.eps + denom)

                                         
    if cfg.clip_percentile is not None:
        p = torch.quantile(pcr.flatten(), cfg.clip_percentile / 100.0)
        pcr = torch.clamp(pcr, 0.0, p)

    scores = compute_pcr_scores(pcr, cfg)
    return pcr, scores, resid_img


def compute_pcr_scores(pcr: torch.Tensor, cfg: PCRConfig) -> Dict[str, float]:
    H, W = pcr.shape
    flat = pcr.flatten()
    mean_all = float(flat.mean().item())

                              
    k = max(1, int(round(cfg.topk * flat.numel())))
    topk_vals, _ = torch.topk(flat, k)
    score_topk = float((topk_vals.mean() / (flat.mean() + cfg.eps)).item())

                               
    y0f, x0f, y1f, x1f = cfg.core_box
    iy0 = int(round(y0f * H))
    ix0 = int(round(x0f * W))
    iy1 = int(round(y1f * H))
    ix1 = int(round(x1f * W))

    core = pcr[iy0:iy1, ix0:ix1]
    perim = pcr.clone()
    perim[iy0:iy1, ix0:ix1] = float("nan")
    periphery_mean = torch.nanmean(perim)
    core_mean = core.mean()
    score_core_ratio = float((core_mean / (periphery_mean + cfg.eps)).item())

                                                                                             
    q95 = torch.quantile(flat, 0.95)
    peak_density = float((pcr > q95).float().mean().item())

    return {
        "pcr_mean": mean_all,
        "pcr_topk_ratio": score_topk,
        "pcr_core_periphery": score_core_ratio,
        "pcr_peak_density": peak_density,
    }
