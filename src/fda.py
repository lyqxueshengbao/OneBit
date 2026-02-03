from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[float, np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class FDAConfig:
    """
    FDA-MIMO minimal configuration (single target).

    Units:
      - f0: Hz
      - c: m/s
      - df: Hz
      - d: m
      - theta: deg (in APIs below)
      - r: m (in APIs below)
    """

    f0: float = 1e9
    c: float = 3e8
    M: int = 10
    N: int = 10
    df: float = 30e3
    d: Optional[float] = None  # default: lambda0/2

    @property
    def lambda0(self) -> float:
        return self.c / self.f0

    @property
    def d_eff(self) -> float:
        return self.lambda0 / 2.0 if self.d is None else float(self.d)

    @property
    def K(self) -> int:
        return int(self.M * self.N)


def steering_vector_torch(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    cfg: FDAConfig,
    *,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Build FDA-MIMO steering vector a(theta,r) (minimal consistent version).

    Args:
      theta_deg: (...,) deg
      r_m: (...,) m, broadcastable with theta_deg
      cfg: FDAConfig
      dtype: complex dtype

    Returns:
      a: (..., K) complex steering vector (Tx-major flatten: m then n)
    """

    device = theta_deg.device
    theta = torch.deg2rad(theta_deg.to(torch.float32))
    r = r_m.to(torch.float32)

    m = torch.arange(cfg.M, device=device, dtype=torch.float32)  # (M,)
    n = torch.arange(cfg.N, device=device, dtype=torch.float32)  # (N,)

    sin_theta = torch.sin(theta)[..., None, None]  # (...,1,1)
    d = torch.tensor(cfg.d_eff, device=device, dtype=torch.float32)
    lambda0 = torch.tensor(cfg.lambda0, device=device, dtype=torch.float32)

    # Angle phase: -2π/λ0 * (m+n)*d*sinθ
    phi_tx = -2.0 * torch.pi * (m[None, :, None] * d / lambda0) * sin_theta  # (...,M,1)
    phi_rx = -2.0 * torch.pi * (n[None, None, :] * d / lambda0) * sin_theta  # (...,1,N)

    # Range phase (FDA): -4π/c * (f0 + m*df) * r
    f_m = torch.tensor(cfg.f0, device=device, dtype=torch.float32) + m * torch.tensor(
        cfg.df, device=device, dtype=torch.float32
    )  # (M,)
    phi_r = (
        -4.0
        * torch.pi
        * (f_m[None, :, None] / torch.tensor(cfg.c, device=device, dtype=torch.float32))
        * r[..., None, None]
    )  # (...,M,1)

    phi = phi_tx + phi_rx + phi_r  # (...,M,N)
    a_mn = torch.exp(1j * phi.to(torch.complex64))  # (...,M,N)
    a = a_mn.reshape(*a_mn.shape[:-2], cfg.K).to(dtype)
    return a


def steering_vector_np(theta_deg: ArrayLike, r_m: ArrayLike, cfg: FDAConfig) -> np.ndarray:
    """
    Numpy version of steering vector.

    Args:
      theta_deg: (...,) deg
      r_m: (...,) m (broadcastable with theta_deg)

    Returns:
      a: (..., K) complex64
    """

    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float32))
    r = np.asarray(r_m, dtype=np.float32)

    m = np.arange(cfg.M, dtype=np.float32)  # (M,)
    n = np.arange(cfg.N, dtype=np.float32)  # (N,)

    sin_theta = np.sin(theta)[..., None, None]
    d = np.float32(cfg.d_eff)
    lambda0 = np.float32(cfg.lambda0)

    phi_tx = -2.0 * np.pi * (m[None, :, None] * d / lambda0) * sin_theta  # (...,M,1)
    phi_rx = -2.0 * np.pi * (n[None, None, :] * d / lambda0) * sin_theta  # (...,1,N)

    f_m = np.float32(cfg.f0) + m * np.float32(cfg.df)  # (M,)
    phi_r = (
        -4.0 * np.pi * (f_m[None, :, None] / np.float32(cfg.c)) * r[..., None, None]
    )  # (...,M,1)

    phi = phi_tx + phi_rx + phi_r
    a_mn = np.exp(1j * phi).astype(np.complex64)
    return a_mn.reshape(*a_mn.shape[:-2], cfg.K)


def simulate_y_torch(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    snr_db: torch.Tensor,
    cfg: FDAConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate full-precision receive vector y = a(theta,r) + w.

    SNR definition (per element):
      SNR = E[|a_k|^2] / E[|w_k|^2]
    Since |a_k|=1 in this minimal model, noise variance is 1/SNR.

    Args:
      theta_deg: (...,) deg
      r_m: (...,) m
      snr_db: (...,) dB, broadcastable

    Returns:
      y: (..., K) complex
      a: (..., K) complex (noise-free)
    """

    if device is None:
        device = theta_deg.device

    theta_deg = theta_deg.to(device)
    r_m = r_m.to(device)
    snr_db = snr_db.to(device).to(torch.float32)

    a = steering_vector_torch(theta_deg, r_m, cfg, dtype=dtype)

    snr_lin = torch.pow(10.0, snr_db / 10.0).clamp_min(eps)
    noise_var = 1.0 / snr_lin  # per complex sample
    sigma = torch.sqrt(noise_var)

    w = (sigma[..., None] / np.sqrt(2.0)) * (
        torch.randn_like(a.real) + 1j * torch.randn_like(a.imag)
    )
    y = a + w.to(dtype)
    return y, a


def simulate_y_np(
    theta_deg: ArrayLike,
    r_m: ArrayLike,
    snr_db: ArrayLike,
    cfg: FDAConfig,
    *,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy version of y = a + w. See simulate_y_torch for SNR definition.

    Returns:
      y: (..., K) complex64
      a: (..., K) complex64
    """

    theta_deg = np.asarray(theta_deg, dtype=np.float32)
    r_m = np.asarray(r_m, dtype=np.float32)
    snr_db = np.asarray(snr_db, dtype=np.float32)

    a = steering_vector_np(theta_deg, r_m, cfg).astype(np.complex64)
    snr_lin = np.maximum(10.0 ** (snr_db / 10.0), eps).astype(np.float32)
    noise_var = 1.0 / snr_lin
    sigma = np.sqrt(noise_var).astype(np.float32)

    w = (sigma[..., None] / np.sqrt(2.0)) * (
        rng.standard_normal(a.shape, dtype=np.float32)
        + 1j * rng.standard_normal(a.shape, dtype=np.float32)
    )
    y = a + w.astype(np.complex64)
    return y.astype(np.complex64), a.astype(np.complex64)

