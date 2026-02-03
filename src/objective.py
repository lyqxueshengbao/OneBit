from __future__ import annotations

import numpy as np
import torch

from .fda import FDAConfig, steering_vector_np, steering_vector_torch


def normalized_correlation_torch(
    a: torch.Tensor, z: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    """
    J = |a^H z|^2 / ||a||^2

    Shapes:
      a: (...,K) complex
      z: (...,K) complex
      out: (...) real
    """

    inner = torch.sum(torch.conj(a) * z, dim=-1)
    num = torch.abs(inner) ** 2
    denom = torch.sum(torch.abs(a) ** 2, dim=-1).clamp_min(eps)
    return (num / denom).real


def normalized_correlation_np(a: np.ndarray, z: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    inner = np.sum(np.conj(a) * z, axis=-1)
    num = np.abs(inner) ** 2
    denom = np.maximum(np.sum(np.abs(a) ** 2, axis=-1), eps)
    return (num / denom).astype(np.float32)


def J_torch(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    z: torch.Tensor,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    a = steering_vector_torch(theta_deg, r_m, cfg, dtype=dtype)
    return normalized_correlation_torch(a, z, eps=eps)


def loglike_logistic(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    z: torch.Tensor,
    beta: float | torch.Tensor,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    1-bit logistic soft-sign log-likelihood (larger is better).

    Model (per sensor, real/imag separately):
      ll_k = log(sigmoid(beta * zR_k * yR_k)) + log(sigmoid(beta * zI_k * yI_k))

    where y = a(theta,r) is complex, and z is 1-bit complex with real/imag in {±1}.

    Shapes:
      theta_deg: (B,) deg
      r_m: (B,) m
      z: (B,K) complex
      out: (B,) real
    """

    a = steering_vector_torch(theta_deg, r_m, cfg, dtype=dtype)  # (B,K)
    yR = a.real.to(torch.float32)
    yI = a.imag.to(torch.float32)

    zR = torch.sign(z.real.to(torch.float32))
    zI = torch.sign(z.imag.to(torch.float32))
    zR = torch.where(zR == 0, torch.ones_like(zR), zR)
    zI = torch.where(zI == 0, torch.ones_like(zI), zI)

    if isinstance(beta, torch.Tensor):
        b = beta.to(device=z.device, dtype=torch.float32)
    else:
        b = torch.tensor(beta, device=z.device, dtype=torch.float32)

    tR = (b * zR * yR).clamp(-50.0, 50.0)
    tI = (b * zI * yI).clamp(-50.0, 50.0)

    ll = torch.nn.functional.logsigmoid(tR) + torch.nn.functional.logsigmoid(tI)
    ll = torch.nan_to_num(ll, nan=-1e6, posinf=0.0, neginf=-1e6)
    return ll.sum(dim=-1)


def loglike_probit(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    z: torch.Tensor,
    beta: float | torch.Tensor,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    1-bit probit soft-sign log-likelihood (larger is better).

    ll = sum_k log(Phi(beta*zR*yR)) + log(Phi(beta*zI*yI))
    Phi(x) = 0.5 * (1 + erf(x/sqrt(2))).
    """

    a = steering_vector_torch(theta_deg, r_m, cfg, dtype=dtype)
    yR = a.real.to(torch.float32)
    yI = a.imag.to(torch.float32)

    zR = torch.sign(z.real.to(torch.float32))
    zI = torch.sign(z.imag.to(torch.float32))
    zR = torch.where(zR == 0, torch.ones_like(zR), zR)
    zI = torch.where(zI == 0, torch.ones_like(zI), zI)

    if isinstance(beta, torch.Tensor):
        b = beta.to(device=z.device, dtype=torch.float32)
    else:
        b = torch.tensor(beta, device=z.device, dtype=torch.float32)

    tR = (b * zR * yR).clamp(-20.0, 20.0)
    tI = (b * zI * yI).clamp(-20.0, 20.0)

    inv_sqrt2 = float(1.0 / np.sqrt(2.0))
    pR = 0.5 * (1.0 + torch.erf(tR * inv_sqrt2))
    pI = 0.5 * (1.0 + torch.erf(tI * inv_sqrt2))
    pR = pR.clamp(eps, 1.0 - eps)
    pI = pI.clamp(eps, 1.0 - eps)

    ll = torch.log(pR) + torch.log(pI)
    ll = torch.nan_to_num(ll, nan=-1e6, posinf=0.0, neginf=-1e6)
    return ll.sum(dim=-1)


def J_np(
    theta_deg: np.ndarray,
    r_m: np.ndarray,
    z: np.ndarray,
    cfg: FDAConfig,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    a = steering_vector_np(theta_deg, r_m, cfg)
    return normalized_correlation_np(a, z, eps=eps)
