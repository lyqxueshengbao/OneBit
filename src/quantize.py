from __future__ import annotations

import numpy as np
import torch


def sign_pm1_torch(x: torch.Tensor, *, sign0: bool = True) -> torch.Tensor:
    """
    sign(x) in {-1,+1} with optional sign0: 0 -> +1.
    """

    s = torch.sign(x)
    if sign0:
        s = torch.where(s == 0, torch.ones_like(s), s)
    return s


def sign_pm1_np(x: np.ndarray, *, sign0: bool = True) -> np.ndarray:
    s = np.sign(x)
    if sign0:
        s = np.where(s == 0, 1.0, s)
    return s.astype(np.float32)


def quantize_1bit_torch(y: torch.Tensor, *, sign0: bool = True) -> torch.Tensor:
    """
    1-bit complex quantization:
      z = sign(Re(y)) + j sign(Im(y))

    Args:
      y: (...,K) complex
    Returns:
      z: (...,K) complex with real/imag in {-1,+1}
    """

    re = sign_pm1_torch(y.real, sign0=sign0)
    im = sign_pm1_torch(y.imag, sign0=sign0)
    return (re + 1j * im).to(torch.complex64)


def quantize_1bit_np(y: np.ndarray, *, sign0: bool = True) -> np.ndarray:
    re = sign_pm1_np(y.real, sign0=sign0)
    im = sign_pm1_np(y.imag, sign0=sign0)
    return (re + 1j * im).astype(np.complex64)

