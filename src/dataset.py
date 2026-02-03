from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .fda import FDAConfig, simulate_y_np, simulate_y_torch
from .quantize import quantize_1bit_np, quantize_1bit_torch


@dataclass(frozen=True)
class TargetBox:
    theta_min: float = -60.0
    theta_max: float = 60.0
    r_min: float = 0.0
    r_max: float = 2000.0


class ParamDataset(Dataset):
    """
    Deterministic dataset of (theta*, r*, snr_db) for reproducibility.

    It does NOT generate y/z in __getitem__ (so you can generate on GPU in the training loop).
    """

    def __init__(
        self,
        num_samples: int,
        *,
        box: TargetBox = TargetBox(),
        snr_range_db: Tuple[float, float] = (-15.0, 15.0),
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.box = box
        self.snr_range_db = snr_range_db
        rng = np.random.default_rng(int(seed))
        self.theta = rng.uniform(box.theta_min, box.theta_max, size=(self.num_samples,)).astype(
            np.float32
        )
        self.r = rng.uniform(box.r_min, box.r_max, size=(self.num_samples,)).astype(np.float32)
        self.snr_db = rng.uniform(snr_range_db[0], snr_range_db[1], size=(self.num_samples,)).astype(
            np.float32
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return {
            "theta_deg": float(self.theta[idx]),
            "r_m": float(self.r[idx]),
            "snr_db": float(self.snr_db[idx]),
        }


def synthesize_batch_torch(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    snr_db: torch.Tensor,
    cfg: FDAConfig,
    *,
    sign0: bool = True,
):
    """
    Generate y and 1-bit z in torch.

    Args:
      theta_deg: (B,) deg
      r_m: (B,) m
      snr_db: (B,) dB

    Returns:
      y: (B,K) complex
      z: (B,K) complex
      a: (B,K) complex (noise-free)
    """

    y, a = simulate_y_torch(theta_deg, r_m, snr_db, cfg)
    z = quantize_1bit_torch(y, sign0=sign0)
    return y, z, a


def synthesize_np(
    theta_deg: np.ndarray,
    r_m: np.ndarray,
    snr_db: np.ndarray,
    cfg: FDAConfig,
    *,
    seed: int = 0,
    sign0: bool = True,
):
    """
    Generate y and 1-bit z in numpy (for CPU baselines / SciPy refinement).

    Notes:
      This function seeds an internal RNG; for Monte Carlo loops you can vary `seed`.
    """

    rng = np.random.default_rng(int(seed))
    y, a = simulate_y_np(theta_deg, r_m, snr_db, cfg, rng=rng)
    z = quantize_1bit_np(y, sign0=sign0)
    return y, z, a

