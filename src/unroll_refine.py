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

    Notes on stability:
      By default this module uses a *first-order* unroll (second_order=False), i.e. we do NOT
      backprop through the gradient operator itself. This dramatically improves stability and
      prevents alpha_raw/lambda_raw from being corrupted by exploding higher-order graphs.
    """

    def __init__(
        self,
        cfg: FDAConfig,
        *,
        T: int = 10,
        box: Box = Box(),
        learnable: bool = True,
        init_alpha: float = 2e-2,
        init_lambda: float = 1e-3,
        alpha_min: float = 1e-5,
        alpha_max: float = 5e-2,
        lambda_min: float = 1e-3,
        lambda_max: float = 1.0,
        step_clip: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.T = int(T)
        self.box = box
        self.eps = float(eps)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.step_clip = float(step_clip)

        # Bounded parameterization via sigmoid:
        #   alpha = a_min + (a_max-a_min) * sigmoid(alpha_raw)
        #   lambda = l_min + (l_max-l_min) * sigmoid(lambda_raw)
        #
        # We initialize raw values by inverse-sigmoid on the normalized interval.
        a0 = torch.full((self.T,), float(init_alpha), dtype=torch.float32)
        l0 = torch.full((self.T,), float(init_lambda), dtype=torch.float32)

        a_p = (a0 - self.alpha_min) / (self.alpha_max - self.alpha_min)
        l_p = (l0 - self.lambda_min) / (self.lambda_max - self.lambda_min)
        alpha_raw = _inv_sigmoid(a_p)
        lambda_raw = _inv_sigmoid(l_p)

        if learnable:
            self.alpha_raw = nn.Parameter(alpha_raw)
            self.lambda_raw = nn.Parameter(lambda_raw)
        else:
            self.register_buffer("alpha_raw", alpha_raw)
            self.register_buffer("lambda_raw", lambda_raw)

    def alpha(self) -> torch.Tensor:
        s = torch.sigmoid(self.alpha_raw)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * s

    def lambd(self) -> torch.Tensor:
        s = torch.sigmoid(self.lambda_raw)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * s

    def forward(
        self,
        z: torch.Tensor,
        theta0_deg: torch.Tensor,
        r0_m: torch.Tensor,
        *,
        T_run: int | None = None,
        second_order: bool = False,
        sanitize_in_forward: bool = False,
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

        # Optional non-inplace sanitization for forward computation only.
        # By default we do NOT modify parameters or silently "fix" them.
        raw_a = self.alpha_raw
        raw_l = self.lambda_raw
        if sanitize_in_forward:
            raw_a = torch.nan_to_num(raw_a, nan=0.0, posinf=20.0, neginf=-20.0)
            raw_l = torch.nan_to_num(raw_l, nan=0.0, posinf=20.0, neginf=-20.0)

        alpha = (self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_a)).to(
            u.device
        )
        lambd = (self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(raw_l)).to(
            u.device
        )

        Js = []
        thetas = []
        rs = []
        nonfinite_g_ratio = []
        fallback_alpha = []
        fallback_lambda = []

        steps = self.T if T_run is None else min(int(T_run), self.T)

        # Always enable grad inside refinement so it works even if caller wraps `torch.no_grad()`.
        with torch.enable_grad():
            for t in range(steps):
                u = u.requires_grad_(True)
                theta, r = map_u_to_theta_r(u, self.box)
                a = steering_vector_torch(theta, r, self.cfg, dtype=z.dtype)
                J = normalized_correlation_torch(a, z)  # (B,)

                g = torch.autograd.grad(
                    (-J).sum(),
                    u,
                    create_graph=bool(second_order),
                    retain_graph=bool(second_order),
                )[0]

                alpha_t = alpha[t]
                lambda_t = lambd[t]

                # Fallback statistics (per-step scalar): whether raw params are non-finite.
                raw_a_t = self.alpha_raw[t]
                raw_l_t = self.lambda_raw[t]
                fallback_alpha.append((~torch.isfinite(raw_a_t)).to(torch.float32).detach())
                fallback_lambda.append((~torch.isfinite(raw_l_t)).to(torch.float32).detach())

                g_finite = torch.isfinite(g)
                nonfinite_g_ratio.append((~g_finite).to(torch.float32).mean().detach())
                g = torch.where(g_finite, g, torch.zeros_like(g))

                denom = torch.abs(g).clamp_min(1e-6) + lambda_t
                denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)

                step_u = alpha_t * g / denom
                step_u = torch.clamp(step_u, -self.step_clip, self.step_clip)
                # Detach the state to prevent unintended graph chaining / memory growth, but keep
                # dependence on alpha/lambda via step_u.
                u = u.detach() - step_u

                if return_trace:
                    Js.append(J.detach())
                    thetas.append(theta.detach())
                    rs.append(r.detach())

        theta_T, r_T = map_u_to_theta_r(u, self.box)
        aT = steering_vector_torch(theta_T, r_T, self.cfg, dtype=z.dtype)
        JT = normalized_correlation_torch(aT, z)

        if not return_trace:
            return theta_T, r_T, JT
        return theta_T, r_T, JT, {
            "J": Js,
            "theta": thetas,
            "r": rs,
            "nonfinite_g_ratio": nonfinite_g_ratio,
            "fallback_alpha": fallback_alpha,
            "fallback_lambda": fallback_lambda,
        }
