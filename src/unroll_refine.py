from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .fda import FDAConfig, steering_vector_torch
from .objective import normalized_correlation_torch


def _inv_sigmoid(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _inv_tanh(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))


@dataclass(frozen=True)
class Box:
    theta_min: float = -60.0
    theta_max: float = 60.0
    r_min: float = 0.0
    r_max: float = 2000.0


def map_u_to_theta_r(u: torch.Tensor, box: Box) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map unconstrained u -> (theta,r) within the valid box via sigmoid.

    Args:
      u: (B,2) (u_theta, u_r)
    Returns:
      theta_deg: (B,)
      r_m: (B,)
    """

    # theta: tanh map (symmetric); r: sigmoid map.
    theta_mid = 0.5 * (box.theta_min + box.theta_max)
    theta_half = 0.5 * (box.theta_max - box.theta_min)
    theta = theta_mid + theta_half * torch.tanh(u[:, 0])

    r = box.r_min + (box.r_max - box.r_min) * torch.sigmoid(u[:, 1])
    return theta, r


def map_theta_r_to_u(theta_deg: torch.Tensor, r_m: torch.Tensor, box: Box) -> torch.Tensor:
    """
    Inverse map (theta,r) -> u via logit.
    """

    theta_mid = 0.5 * (box.theta_min + box.theta_max)
    theta_half = 0.5 * (box.theta_max - box.theta_min)
    y_theta = (theta_deg - theta_mid) / theta_half

    p_r = (r_m - box.r_min) / (box.r_max - box.r_min)
    return torch.stack([_inv_tanh(y_theta), _inv_sigmoid(p_r)], dim=-1)


class Refiner(nn.Module):
    """
    Unrolled off-grid refinement module under the same objective J(theta,r).

    Update rule (per step t):
      u <- u - alpha_t * g / (|g| + lambda_t)
    where g = ∂(-J)/∂u from autograd.
    """

    def __init__(
        self,
        cfg: FDAConfig,
        *,
        T: int = 10,
        box: Box = Box(),
        learnable: bool = True,
        init_alpha: float = 0.1,
        init_lambda: float = 1e-3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.T = int(T)
        self.box = box
        self.eps = float(eps)

        alpha0 = torch.full((self.T,), float(init_alpha))
        lambda0 = torch.full((self.T,), float(init_lambda))

        # Parameterize with softplus to keep positivity.
        alpha_raw = torch.log(torch.expm1(alpha0))
        lambda_raw = torch.log(torch.expm1(lambda0))

        if learnable:
            self.alpha_raw = nn.Parameter(alpha_raw)
            self.lambda_raw = nn.Parameter(lambda_raw)
        else:
            self.register_buffer("alpha_raw", alpha_raw)
            self.register_buffer("lambda_raw", lambda_raw)

    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_raw) + self.eps

    def lambd(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.lambda_raw) + self.eps

    def forward(
        self,
        z: torch.Tensor,
        theta0_deg: torch.Tensor,
        r0_m: torch.Tensor,
        *,
        T_run: int | None = None,
        return_trace: bool = False,
    ):
        """
        Args:
          z: (B,K) complex64
          theta0_deg: (B,) deg (coarse init)
          r0_m: (B,) m (coarse init)
        Returns:
          theta_T: (B,) deg
          r_T: (B,) m
          J_T: (B,) objective value at final step
          trace (optional): dict of intermediate tensors (detached)
        """

        u = map_theta_r_to_u(theta0_deg, r0_m, self.box)
        alpha = self.alpha().to(u.device)
        lambd = self.lambd().to(u.device)

        Js = []
        thetas = []
        rs = []

        steps = self.T if T_run is None else min(int(T_run), self.T)

        for t in range(steps):
            u = u.requires_grad_(True)
            theta, r = map_u_to_theta_r(u, self.box)
            a = steering_vector_torch(theta, r, self.cfg, dtype=z.dtype)
            J = normalized_correlation_torch(a, z)  # (B,)

            g = torch.autograd.grad(
                (-J).sum(), u, create_graph=self.training, retain_graph=self.training
            )[0]
            u = u - alpha[t] * g / (torch.abs(g) + lambd[t])

            if return_trace:
                Js.append(J.detach())
                thetas.append(theta.detach())
                rs.append(r.detach())

        theta_T, r_T = map_u_to_theta_r(u, self.box)
        aT = steering_vector_torch(theta_T, r_T, self.cfg, dtype=z.dtype)
        JT = normalized_correlation_torch(aT, z)

        if not return_trace:
            return theta_T, r_T, JT
        return theta_T, r_T, JT, {"J": Js, "theta": thetas, "r": rs}
