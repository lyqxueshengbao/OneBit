from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_all import run_unrolled
from src.coarse_search import coarse_search_np
from src.dataset import TargetBox, synthesize_np
from src.fda import FDAConfig
from src.metrics import angle_error_deg_np, rmse_np
from src.nm_refine import refine_nelder_mead
from src.objective import J_np
from src.utils import CsvLogger, Timer, ensure_dir, seed_all, timestamp, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snr_list", type=str, default="0,5,10,15,20")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--nm_maxiter", type=int, default=60)
    p.add_argument("--T", type=int, default=1)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--sanitize_ckpt", action="store_true")
    p.add_argument("--full", action="store_true")

    # Local INR refine config (coarse -> local INR).
    p.add_argument("--inr_num_samples", type=int, default=128)
    p.add_argument("--inr_window_theta_deg", type=float, default=4.0)
    p.add_argument("--inr_window_r_m", type=float, default=200.0)
    p.add_argument("--inr_train_points", type=int, default=128)
    p.add_argument("--inr_steps", type=int, default=200)
    p.add_argument("--inr_lr", type=float, default=1e-3)
    p.add_argument("--inr_width", type=int, default=64)
    p.add_argument("--inr_depth", type=int, default=3)
    p.add_argument("--inr_eval_theta_step", type=float, default=0.25)
    p.add_argument("--inr_eval_r_step", type=float, default=10.0)
    p.add_argument("--inr_use_phase_pe", type=int, default=1, choices=[0, 1])
    p.add_argument("--inr_pe_theta_terms", type=int, default=8)
    p.add_argument("--inr_pe_r_terms", type=int, default=8)
    p.add_argument("--inr_nm_tail_iters", type=int, default=0)
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def ms_per_sample(dt_s: float, n: int) -> float:
    return 1000.0 * dt_s / max(int(n), 1)


class LocalINR(torch.nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int) -> None:
        super().__init__()
        depth = max(int(depth), 2)
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, int(width)), torch.nn.SiLU()]
        for _ in range(depth - 2):
            layers.append(torch.nn.Linear(int(width), int(width)))
            layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Linear(int(width), 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_local_range(
    theta0: float,
    r0: float,
    box: TargetBox,
    win_theta: float,
    win_r: float,
) -> tuple[float, float, float, float]:
    t_lo = max(float(box.theta_min), float(theta0) - float(win_theta))
    t_hi = min(float(box.theta_max), float(theta0) + float(win_theta))
    r_lo = max(float(box.r_min), float(r0) - float(win_r))
    r_hi = min(float(box.r_max), float(r0) + float(win_r))
    if t_hi <= t_lo:
        t_lo, t_hi = float(theta0) - 0.5, float(theta0) + 0.5
    if r_hi <= r_lo:
        r_lo, r_hi = max(float(box.r_min), float(r0) - 1.0), min(float(box.r_max), float(r0) + 1.0)
    return t_lo, t_hi, r_lo, r_hi


def _phase_pe(
    theta_deg: torch.Tensor,
    r_m: torch.Tensor,
    cfg: FDAConfig,
    *,
    use_phase_pe: bool,
    pe_theta_terms: int,
    pe_r_terms: int,
    box: TargetBox,
) -> torch.Tensor:
    theta_rad = torch.deg2rad(theta_deg.to(torch.float32))
    r_m = r_m.to(torch.float32)

    # Base coordinates.
    t_norm = (theta_deg.to(torch.float32) - 0.5 * (box.theta_min + box.theta_max)) / (
        0.5 * (box.theta_max - box.theta_min) + 1e-12
    )
    r_norm = (r_m - 0.5 * (box.r_min + box.r_max)) / (0.5 * (box.r_max - box.r_min) + 1e-12)
    feats = [t_norm, r_norm, torch.sin(theta_rad), torch.cos(theta_rad)]

    if not bool(use_phase_pe):
        return torch.stack(feats, dim=-1)

    device = theta_deg.device
    dtype = theta_deg.dtype if theta_deg.dtype.is_floating_point else torch.float32

    # FDA-aware phase encodings.
    d_over_lambda = float(cfg.d_eff / cfg.lambda0)
    k_theta = torch.linspace(
        0.0,
        float(cfg.M + cfg.N - 2),
        steps=max(int(pe_theta_terms), 1),
        device=device,
        dtype=dtype,
    )
    phase_theta = (2.0 * np.pi * d_over_lambda) * torch.sin(theta_rad)[..., None] * k_theta[None, :]
    feats.append(torch.sin(phase_theta))
    feats.append(torch.cos(phase_theta))

    k_r = torch.linspace(
        0.0,
        float(cfg.M - 1),
        steps=max(int(pe_r_terms), 1),
        device=device,
        dtype=dtype,
    )
    f_r = float(cfg.f0) + k_r * float(cfg.df)
    phase_r = (4.0 * np.pi / float(cfg.c)) * r_m[..., None] * f_r[None, :]
    feats.append(torch.sin(phase_r))
    feats.append(torch.cos(phase_r))

    out = []
    for f in feats:
        if f.ndim == 1:
            out.append(f.unsqueeze(-1))
        else:
            out.append(f)
    return torch.cat(out, dim=-1)


def _fit_local_inr_single(
    z_i: np.ndarray,
    theta0: float,
    r0: float,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    device: torch.device,
    rng: np.random.Generator,
    inr_window_theta_deg: float,
    inr_window_r_m: float,
    inr_train_points: int,
    inr_steps: int,
    inr_lr: float,
    inr_width: int,
    inr_depth: int,
    inr_eval_theta_step: float,
    inr_eval_r_step: float,
    inr_use_phase_pe: bool,
    inr_pe_theta_terms: int,
    inr_pe_r_terms: int,
) -> tuple[float, float, float]:
    t_lo, t_hi, r_lo, r_hi = _make_local_range(
        float(theta0),
        float(r0),
        box,
        float(inr_window_theta_deg),
        float(inr_window_r_m),
    )

    n_train = max(int(inr_train_points), 16)
    th_train = rng.uniform(t_lo, t_hi, size=(n_train,)).astype(np.float32)
    rr_train = rng.uniform(r_lo, r_hi, size=(n_train,)).astype(np.float32)
    # Always include coarse center as anchor.
    th_train[0] = np.float32(theta0)
    rr_train[0] = np.float32(r0)

    y_train = J_np(th_train, rr_train, z_i, cfg).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-6)
    y_train_n = (y_train - y_mean) / y_std

    th_t = torch.from_numpy(th_train).to(device=device, dtype=torch.float32)
    rr_t = torch.from_numpy(rr_train).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y_train_n).to(device=device, dtype=torch.float32)

    x_t = _phase_pe(
        th_t,
        rr_t,
        cfg,
        use_phase_pe=bool(inr_use_phase_pe),
        pe_theta_terms=int(inr_pe_theta_terms),
        pe_r_terms=int(inr_pe_r_terms),
        box=box,
    )
    model = LocalINR(in_dim=int(x_t.shape[-1]), width=int(inr_width), depth=int(inr_depth)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(inr_lr))

    model.train()
    for _ in range(max(int(inr_steps), 1)):
        pred = model(x_t)
        loss = F.mse_loss(pred, y_t)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    fit_loss = float(loss.detach().cpu().item())

    # Dense local scan on INR.
    th_eval = np.arange(t_lo, t_hi + 1e-6, float(inr_eval_theta_step), dtype=np.float32)
    rr_eval = np.arange(r_lo, r_hi + 1e-6, float(inr_eval_r_step), dtype=np.float32)
    tt, rr = np.meshgrid(th_eval, rr_eval, indexing="ij")
    th_flat = torch.from_numpy(tt.reshape(-1)).to(device=device, dtype=torch.float32)
    rr_flat = torch.from_numpy(rr.reshape(-1)).to(device=device, dtype=torch.float32)
    x_eval = _phase_pe(
        th_flat,
        rr_flat,
        cfg,
        use_phase_pe=bool(inr_use_phase_pe),
        pe_theta_terms=int(inr_pe_theta_terms),
        pe_r_terms=int(inr_pe_r_terms),
        box=box,
    )
    model.eval()
    with torch.no_grad():
        pred_eval = model(x_eval).detach().cpu().numpy().astype(np.float32)
    pred_eval = pred_eval * y_std + y_mean
    idx = int(np.argmax(pred_eval))
    return float(th_flat[idx].item()), float(rr_flat[idx].item()), fit_loss


def run_inr_local_refine(
    z: np.ndarray,
    theta0: np.ndarray,
    r0: np.ndarray,
    cfg: FDAConfig,
    box: TargetBox,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_total = int(z.shape[0])
    n_eval = int(args.inr_num_samples) if int(args.inr_num_samples) > 0 else n_total
    n_eval = min(n_total, n_eval)

    theta_hat = np.zeros((n_eval,), dtype=np.float32)
    r_hat = np.zeros((n_eval,), dtype=np.float32)
    fit_losses = []
    rng = np.random.default_rng(int(args.seed) + 20260209)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_eval):
        th, rr, l = _fit_local_inr_single(
            z[i],
            float(theta0[i]),
            float(r0[i]),
            cfg,
            box,
            device=device,
            rng=rng,
            inr_window_theta_deg=float(args.inr_window_theta_deg),
            inr_window_r_m=float(args.inr_window_r_m),
            inr_train_points=int(args.inr_train_points),
            inr_steps=int(args.inr_steps),
            inr_lr=float(args.inr_lr),
            inr_width=int(args.inr_width),
            inr_depth=int(args.inr_depth),
            inr_eval_theta_step=float(args.inr_eval_theta_step),
            inr_eval_r_step=float(args.inr_eval_r_step),
            inr_use_phase_pe=bool(int(args.inr_use_phase_pe)),
            inr_pe_theta_terms=int(args.inr_pe_theta_terms),
            inr_pe_r_terms=int(args.inr_pe_r_terms),
        )
        theta_hat[i] = np.float32(th)
        r_hat[i] = np.float32(rr)
        fit_losses.append(float(l))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return theta_hat, r_hat, ms_per_sample(dt, n_eval), float(np.mean(fit_losses) if fit_losses else 0.0)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.full:
        # INR is per-sample inner-loop fitting; keep full mode practical.
        args.num_samples = max(int(args.num_samples), 512)
        args.nm_maxiter = max(int(args.nm_maxiter), 120)
        if int(args.inr_num_samples) <= 0:
            args.inr_num_samples = min(int(args.num_samples), 256)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()
    t_run = int(args.T)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"eval_inr_{timestamp()}"
    ensure_dir(run_dir)
    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "device_used": str(device),
            "cfg": cfg.__dict__,
        },
    )

    csv = CsvLogger(
        run_dir / "results.csv",
        fieldnames=["snr_db", "method", "rmse_theta_deg", "rmse_r_m", "ms_per_sample", "notes"],
    )

    snr_list = [float(x) for x in args.snr_list.split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))

    for snr_db in snr_list:
        theta_gt = rng.uniform(box.theta_min, box.theta_max, size=(int(args.num_samples),)).astype(np.float32)
        r_gt = rng.uniform(box.r_min, box.r_max, size=(int(args.num_samples),)).astype(np.float32)
        snr_vec = np.full((int(args.num_samples),), float(snr_db), dtype=np.float32)
        _, z, _ = synthesize_np(theta_gt, r_gt, snr_vec, cfg, seed=args.seed + 34567)
        z = z.astype(np.complex64)

        # 1) grid-only baseline.
        theta_grid = np.zeros_like(theta_gt)
        r_grid = np.zeros_like(r_gt)
        with Timer() as t_grid:
            for i in range(int(args.num_samples)):
                th0, rr0, _ = coarse_search_np(
                    z[i],
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    theta_step=float(args.theta_step),
                    r_step=float(args.r_step),
                )
                theta_grid[i] = th0
                r_grid[i] = rr0
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_only_cpu",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(theta_grid, theta_gt)),
                "rmse_r_m": rmse_np(r_grid - r_gt),
                "ms_per_sample": ms_per_sample(t_grid.dt, int(args.num_samples)),
                "notes": "",
            }
        )

        # 2) grid + NM baseline.
        theta_nm = np.zeros_like(theta_gt)
        r_nm = np.zeros_like(r_gt)
        with Timer() as t_nm:
            for i in range(int(args.num_samples)):
                res = refine_nelder_mead(
                    z[i],
                    float(theta_grid[i]),
                    float(r_grid[i]),
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    maxiter=int(args.nm_maxiter),
                )
                theta_nm[i] = np.float32(res.theta_deg)
                r_nm[i] = np.float32(res.r_m)
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_nm_cpu",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(theta_nm, theta_gt)),
                "rmse_r_m": rmse_np(r_nm - r_gt),
                "ms_per_sample": ms_per_sample(t_grid.dt + t_nm.dt, int(args.num_samples)),
                "notes": f"nm_only_ms={ms_per_sample(t_nm.dt, int(args.num_samples)):.3f}",
            }
        )

        # 3) grid + unroll fixed.
        if t_run <= 0:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_fixed",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": f"skip: T_run={t_run} (no unroll steps)",
                }
            )
        else:
            th_u, rr_u, ms_u = run_unrolled(
                z,
                cfg,
                box,
                device=device,
                theta_step=float(args.theta_step),
                r_step=float(args.r_step),
                T=t_run,
                ckpt_path="",
                sanitize_ckpt=False,
                learnable=False,
                T_run=t_run,
                batch_size=int(args.batch_size),
            )
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_fixed",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(th_u, theta_gt)),
                    "rmse_r_m": rmse_np(rr_u - r_gt),
                    "ms_per_sample": ms_u,
                    "notes": f"device={device.type}; T_run={t_run}",
                }
            )

        # 4) optional learned unroll.
        if args.ckpt_path and t_run > 0:
            try:
                th_l, rr_l, ms_l, t_model_l, t_run_l = run_unrolled(
                    z,
                    cfg,
                    box,
                    device=device,
                    theta_step=float(args.theta_step),
                    r_step=float(args.r_step),
                    T=t_run,
                    ckpt_path=str(args.ckpt_path),
                    sanitize_ckpt=bool(args.sanitize_ckpt),
                    learnable=True,
                    T_run=t_run,
                    return_meta=True,
                    batch_size=int(args.batch_size),
                )
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned",
                        "rmse_theta_deg": rmse_np(angle_error_deg_np(th_l, theta_gt)),
                        "rmse_r_m": rmse_np(rr_l - r_gt),
                        "ms_per_sample": ms_l,
                        "notes": f"ckpt={args.ckpt_path}; T_model={t_model_l}, T_run={t_run_l}",
                    }
                )
            except RuntimeError as e:
                csv.log(
                    {
                        "snr_db": snr_db,
                        "method": "grid_unroll_learned",
                        "rmse_theta_deg": "",
                        "rmse_r_m": "",
                        "ms_per_sample": "",
                        "notes": f"skip ({str(e)})",
                    }
                )
        else:
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": "grid_unroll_learned",
                    "rmse_theta_deg": "",
                    "rmse_r_m": "",
                    "ms_per_sample": "",
                    "notes": "skip (no --ckpt_path or T_run<=0)",
                }
            )

        # 5) coarse -> local INR refine (subset by inr_num_samples).
        th_inr, rr_inr, ms_inr, fit_loss = run_inr_local_refine(
            z,
            theta_grid,
            r_grid,
            cfg,
            box,
            args=args,
            device=device,
        )
        n_eval = int(th_inr.shape[0])
        csv.log(
            {
                "snr_db": snr_db,
                "method": "grid_inr_local",
                "rmse_theta_deg": rmse_np(angle_error_deg_np(th_inr, theta_gt[:n_eval])),
                "rmse_r_m": rmse_np(rr_inr - r_gt[:n_eval]),
                "ms_per_sample": ms_inr,
                "notes": (
                    f"n_eval={n_eval}; use_phase_pe={int(args.inr_use_phase_pe)}; "
                    f"win=({float(args.inr_window_theta_deg):.2f}deg,{float(args.inr_window_r_m):.1f}m); "
                    f"train_pts={int(args.inr_train_points)}; steps={int(args.inr_steps)}; "
                    f"fit_mse={fit_loss:.3e}"
                ),
            }
        )

        # 6) optional tiny NM tail after INR.
        if int(args.inr_nm_tail_iters) > 0:
            th_tail = np.zeros_like(th_inr)
            rr_tail = np.zeros_like(rr_inr)
            with Timer() as t_tail:
                for i in range(n_eval):
                    res = refine_nelder_mead(
                        z[i],
                        float(th_inr[i]),
                        float(rr_inr[i]),
                        cfg,
                        theta_range=(box.theta_min, box.theta_max),
                        r_range=(box.r_min, box.r_max),
                        maxiter=int(args.inr_nm_tail_iters),
                    )
                    th_tail[i] = np.float32(res.theta_deg)
                    rr_tail[i] = np.float32(res.r_m)
            tail_ms = ms_per_sample(t_tail.dt, n_eval)
            csv.log(
                {
                    "snr_db": snr_db,
                    "method": f"grid_inr_local_nm{int(args.inr_nm_tail_iters)}",
                    "rmse_theta_deg": rmse_np(angle_error_deg_np(th_tail, theta_gt[:n_eval])),
                    "rmse_r_m": rmse_np(rr_tail - r_gt[:n_eval]),
                    "ms_per_sample": float(ms_inr) + float(tail_ms),
                    "notes": f"n_eval={n_eval}; nm_tail_only_ms={tail_ms:.3f}",
                }
            )

        print(f"SNR={snr_db} done")

    print(f"Wrote: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
