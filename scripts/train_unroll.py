from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Allow both `python -m scripts.train_unroll` and `python scripts/train_unroll.py`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.coarse_search import CoarseSearcherTorch
from src.dataset import TargetBox, synthesize_batch_torch
from src.fda import FDAConfig
from src.metrics import angle_error_deg_torch
from src.nm_refine import refine_nelder_mead, refine_nelder_mead_with_history
from src.unroll_refine import Box, Refiner, map_theta_r_to_u, map_u_to_theta_r
from src.utils import JsonlLogger, ensure_dir, seed_all, timestamp, write_json


R0 = 50.0


def parse_pscale_input_tokens(s: str) -> list[str]:
    allowed = {"step", "u", "t"}
    raw = [tok.strip().lower() for tok in str(s).split(",")]
    tokens: list[str] = []
    seen = set()
    for tok in raw:
        if not tok:
            continue
        if tok not in allowed:
            raise ValueError(
                f"Invalid --pscale_input token: {tok!r}. Allowed tokens: {sorted(allowed)}. "
                f"Got input: {s!r}"
            )
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)
    if not tokens:
        raise ValueError(
            f"Empty --pscale_input after parsing. Allowed tokens: {sorted(allowed)}. Got input: {s!r}"
        )
    return tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=10)
    # Total optimization steps. If omitted, can be derived from --epochs * --steps_per_epoch (compat).
    p.add_argument("--steps", type=int, default=None)
    # Backward-compatible epoch-style interface (optional).
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    # Global grad norm clipping (disabled by default; clip threshold in L2 norm).
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--snr_min", type=float, default=-15.0)
    p.add_argument("--snr_max", type=float, default=15.0)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=5)
    # Early stopping (based on validation metrics)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_score",
        choices=["val_score", "theta", "r"],
    )
    # Likelihood sharpness schedule
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta_final", type=float, default=3.0)
    p.add_argument("--beta_warmup_frac", type=float, default=0.3)
    # NM teacher distillation
    p.add_argument("--teacher", type=str, default="nm", choices=["none", "nm"])
    # Backward-compatible alias for --teacher
    p.add_argument("--kd_mode", type=str, default=None, choices=["none", "nm"])
    p.add_argument("--teacher_prob", type=float, default=1.0)
    p.add_argument("--teacher_snr_min", type=float, default=0.0)
    p.add_argument("--teacher_maxiter", type=int, default=60)
    p.add_argument("--w_nm", type=float, default=1.0)
    # Backward-compatible alias for --w_nm (teacher weight)
    p.add_argument("--w_teacher", type=float, default=None)
    # Step-wise teacher distillation (NM trajectory)
    p.add_argument("--w_kd_step", type=float, default=1.0)
    p.add_argument("--w_kd_step_theta", type=float, default=0.2)
    p.add_argument("--w_kd_step_r", type=float, default=1.0)
    p.add_argument(
        "--kd_step_t_weight",
        type=str,
        default="uniform",
        choices=["uniform", "late"],
    )
    p.add_argument(
        "--kd_step_mode",
        type=str,
        default="abs",
        choices=["abs", "delta_u", "delta_tr"],
    )
    p.add_argument("--w_gt", type=float, default=0.2)
    p.add_argument("--w_phys", type=float, default=0.05)
    # Optional safety cap for nm_lambda_eff schedule.
    p.add_argument("--nm_lambda_eff_max", type=float, default=None)
    # Normalized GT loss (theta/r joint optimization)
    p.add_argument("--theta_scale", type=float, default=10.0)
    p.add_argument("--r_scale", type=float, default=2000.0)
    p.add_argument("--w_theta", type=float, default=1.0)
    p.add_argument("--w_r", type=float, default=1.0)
    p.add_argument("--huber_delta", type=float, default=1.0)
    # Step-wise KD time weighting (alias for --kd_step_t_weight)
    p.add_argument("--kd_time_weight", type=str, default=None, choices=["uniform", "late"])
    # Optional debug trace dumping (off by default)
    p.add_argument("--dump_trace_every", type=int, default=0)
    # r precondition scaling (only affects r update when refiner.r_box == "tanh")
    p.add_argument("--r_precond_mul", type=float, default=1.0)
    p.add_argument("--r_precond_pow", type=float, default=1.0)
    p.add_argument("--r_precond_learnable", type=int, default=0, choices=[0, 1])
    # Tiny per-sample scale preconditioner (pscale)
    p.add_argument("--use_pscale", type=int, default=0, choices=[0, 1])
    p.add_argument("--pscale_hidden", type=int, default=32)
    p.add_argument("--pscale_detach_step", type=int, default=1, choices=[0, 1])
    p.add_argument("--pscale_logrange", type=float, default=6.9)
    p.add_argument("--pscale_amp", type=float, default=1.0)
    p.add_argument(
        "--pscale_input",
        type=str,
        default="step,u,t",
        help='Comma-separated tokens for pscale features. Supported: "step,u,t" (order matters; deduped).',
    )
    p.add_argument("--use_t_table", type=int, default=0, choices=[0, 1])
    p.add_argument("--t_table_init", type=float, default=0.0)
    p.add_argument("--pscale_min_theta", type=float, default=0.7)
    p.add_argument("--pscale_max_theta", type=float, default=1.3)
    p.add_argument("--pscale_min_r", type=float, default=0.7)
    p.add_argument("--pscale_max_r", type=float, default=1.3)
    p.add_argument("--pscale_reg_w", type=float, default=1e-3)
    p.add_argument("--t_table_freeze_steps", type=int, default=2000)
    p.add_argument("--t_table_lr_mult", type=float, default=0.1)
    p.add_argument("--kd_warmup_steps", type=int, default=1500)
    p.add_argument("--nm_ramp_steps", type=int, default=1500)
    # Backward-compatible name -> run_dir (runs/<run_name>)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def mse_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x * x)


def huber_to_zero(x: torch.Tensor, *, delta: float) -> torch.Tensor:
    return torch.nn.functional.huber_loss(
        x.to(torch.float32),
        torch.zeros_like(x, dtype=torch.float32),
        delta=float(delta),
        reduction="mean",
    ).to(x.dtype)


def masked_mse(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if mask.numel() == 0 or not mask.any().item():
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return mse_to_zero(x[mask])


def masked_pair_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mse(x, mask) + masked_mse(y, mask)


def total_grad_norm_l2(parameters) -> torch.Tensor:
    grads = []
    for p in parameters:
        if p is None:
            continue
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if g.is_sparse:
            g = g.coalesce().values()
        grads.append(torch.norm(g.detach(), p=2))
    if not grads:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(grads), ord=2)


def gt_loss(
    theta_err_deg: torch.Tensor,
    r_err_m: torch.Tensor,
    *,
    theta_scale: float,
    r_scale: float,
    w_theta: float,
    w_r: float,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_n = theta_err_deg / float(theta_scale)
    r_n = r_err_m / float(r_scale)
    loss_theta = huber_to_zero(theta_n, delta=float(huber_delta))
    loss_r = huber_to_zero(r_n, delta=float(huber_delta))
    loss = float(w_theta) * loss_theta + float(w_r) * loss_r
    return loss, loss_theta, loss_r


def masked_gt_loss(
    theta_err_deg: torch.Tensor,
    r_err_m: torch.Tensor,
    mask: torch.Tensor,
    *,
    theta_scale: float,
    r_scale: float,
    w_theta: float,
    w_r: float,
    huber_delta: float,
) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if mask.numel() == 0 or not mask.any().item():
        return torch.tensor(0.0, device=theta_err_deg.device, dtype=theta_err_deg.dtype)
    loss, _, _ = gt_loss(
        theta_err_deg[mask],
        r_err_m[mask],
        theta_scale=theta_scale,
        r_scale=r_scale,
        w_theta=w_theta,
        w_r=w_r,
        huber_delta=huber_delta,
    )
    return loss


def t_weights(T: int, mode: str, device: torch.device) -> torch.Tensor:
    if T <= 0:
        raise ValueError("T must be positive")
    if mode == "uniform":
        w = torch.ones((T,), device=device, dtype=torch.float32)
    elif mode == "late":
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32) / float(T)
        w = t * t
    else:
        raise ValueError(f"Unknown kd_time_weight: {mode}")
    w = w / w.sum().clamp_min(1e-12)
    return w


def beta_schedule(step: int, total_steps: int, beta0: float, beta1: float, warmup_frac: float) -> float:
    warmup_steps = max(1, int(float(total_steps) * float(warmup_frac)))
    t = min(float(step) / float(warmup_steps), 1.0)
    return float(beta0 + (beta1 - beta0) * t)


@torch.no_grad()
def sample_gt(batch: int, box: TargetBox, snr_min: float, snr_max: float, device: torch.device):
    theta = (box.theta_min + (box.theta_max - box.theta_min) * torch.rand(batch, device=device)).to(
        torch.float32
    )
    r = (box.r_min + (box.r_max - box.r_min) * torch.rand(batch, device=device)).to(torch.float32)
    snr = (snr_min + (snr_max - snr_min) * torch.rand(batch, device=device)).to(torch.float32)
    return theta, r, snr


def main() -> None:
    args = parse_args()
    if args.w_teacher is not None:
        args.w_nm = float(args.w_teacher)

    # Backward-compat arg mappings.
    if args.kd_mode is not None:
        args.teacher = str(args.kd_mode)
    if args.steps is None:
        if args.epochs is not None or args.steps_per_epoch is not None:
            if args.epochs is None or args.steps_per_epoch is None:
                raise ValueError("--epochs and --steps_per_epoch must be provided together when --steps is omitted.")
            args.steps = int(args.epochs) * int(args.steps_per_epoch)
        else:
            args.steps = 2000
    args.steps = int(args.steps)

    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()

    if int(args.use_pscale) == 1:
        pscale_tokens = parse_pscale_input_tokens(args.pscale_input)
        args.pscale_input = ",".join(pscale_tokens)
        print(f"[pscale] input_tokens={pscale_tokens} feat_dim={len(pscale_tokens)}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_name:
        run_dir = Path("runs") / str(args.run_name)
    else:
        run_dir = Path("runs") / f"train_{timestamp()}"
    ensure_dir(run_dir)
    write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "cfg": cfg.__dict__,
        },
    )
    logger = JsonlLogger(run_dir / "train_log.jsonl")

    coarse = CoarseSearcherTorch(
        cfg,
        theta_range=(box.theta_min, box.theta_max),
        r_range=(box.r_min, box.r_max),
        theta_step=args.theta_step,
        r_step=args.r_step,
        device=device,
    )

    refiner = Refiner(
        cfg,
        T=args.T,
        box=Box(box.theta_min, box.theta_max, box.r_min, box.r_max),
        learnable=True,
        r_precond_mul=float(args.r_precond_mul),
        r_precond_pow=float(args.r_precond_pow),
        r_precond_learnable=bool(int(args.r_precond_learnable)),
        use_pscale=bool(int(args.use_pscale)),
        pscale_hidden=int(args.pscale_hidden),
        pscale_detach_step=bool(int(args.pscale_detach_step)),
        pscale_logrange=float(args.pscale_logrange),
        pscale_amp=float(args.pscale_amp),
        pscale_input=str(args.pscale_input),
        use_t_table=bool(int(args.use_t_table)),
        t_table_init=float(args.t_table_init),
        pscale_min_theta=float(args.pscale_min_theta),
        pscale_max_theta=float(args.pscale_max_theta),
        pscale_min_r=float(args.pscale_min_r),
        pscale_max_r=float(args.pscale_max_r),
    ).to(device)

    # Optimizer with optional separate param group for t_table.
    t_table = getattr(refiner, "t_log_scale_table", None)
    if t_table is None:
        opt = torch.optim.Adam(refiner.parameters(), lr=args.lr)
    else:
        other_params = [p for p in refiner.parameters() if p is not t_table]
        opt = torch.optim.Adam(
            [
                {"params": other_params, "lr": float(args.lr)},
                {"params": [t_table], "lr": float(args.lr) * float(args.t_table_lr_mult)},
            ]
        )
    last_good_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
    last_good_step = 0
    best_metric = float("inf")
    bad_epochs = 0
    t_table_is_frozen = False

    for step in range(1, args.steps + 1):
        # nm/likelihood/physics term schedule.
        kd_warm = max(int(args.kd_warmup_steps), 0)
        ramp = max(int(args.nm_ramp_steps), 0)
        if step <= kd_warm:
            nm_lambda_eff_raw = 0.0
        elif ramp <= 0:
            nm_lambda_eff_raw = float(args.w_phys)
        else:
            tt = min(step - kd_warm, ramp) / float(ramp)
            nm_lambda_eff_raw = float(args.w_phys) * float(tt)

        nm_lambda_eff_used = float(nm_lambda_eff_raw)
        if args.nm_lambda_eff_max is not None:
            cap = float(args.nm_lambda_eff_max)
            if cap == 0.0:
                nm_lambda_eff_used = 0.0
            elif cap > 0.0:
                nm_lambda_eff_used = float(min(float(nm_lambda_eff_used), cap))

        # Freeze/unfreeze t_table if present.
        t_table = getattr(refiner, "t_log_scale_table", None)
        if t_table is not None and int(args.t_table_freeze_steps) > 0:
            should_freeze = step <= int(args.t_table_freeze_steps)
            if should_freeze and not t_table_is_frozen:
                t_table.requires_grad_(False)
                t_table_is_frozen = True
            if (not should_freeze) and t_table_is_frozen:
                t_table.requires_grad_(True)
                t_table_is_frozen = False

        beta = beta_schedule(step, args.steps, args.beta, args.beta_final, args.beta_warmup_frac)
        theta_gt, r_gt, snr_db = sample_gt(
            args.batch_size, box, args.snr_min, args.snr_max, device
        )
        _, z, _ = synthesize_batch_torch(theta_gt, r_gt, snr_db, cfg)

        theta0, r0, _ = coarse.search(z)
        pscale_step_norm = float(step - 1) / float(max(int(args.steps) - 1, 1))
        kd_time_weight = str(args.kd_time_weight) if args.kd_time_weight is not None else str(args.kd_step_t_weight)
        need_kd_step = (
            args.teacher == "nm"
            and float(args.teacher_prob) > 0.0
            and float(args.w_kd_step) > 0.0
            and (float(args.w_kd_step_theta) > 0.0 or float(args.w_kd_step_r) > 0.0)
        )
        dump_every = int(args.dump_trace_every)
        dump_this_step = dump_every > 0 and (step % dump_every == 0)
        want_trace = bool(need_kd_step or dump_this_step)
        if want_trace:
            theta_hat, r_hat, dbg, trace = refiner(
                z,
                theta0,
                r0,
                beta=beta,
                return_trace=True,
                trace_detach=not bool(need_kd_step),
                pscale_step_norm=pscale_step_norm,
            )
        else:
            theta_hat, r_hat, dbg = refiner(z, theta0, r0, beta=beta, pscale_step_norm=pscale_step_norm)
            trace = None

        theta_err = angle_error_deg_torch(theta_hat, theta_gt)
        r_err = r_hat - r_gt

        # Distill to NM (on a subset for speed) + a small GT term + physics (-ll) regularizer.
        loss_nm = torch.tensor(0.0, device=device)
        loss_nm_snr_ge0 = torch.tensor(0.0, device=device)
        loss_nm_snr_lt0 = torch.tensor(0.0, device=device)
        loss_kd_step = torch.tensor(0.0, device=device)
        loss_kd_step_theta = torch.tensor(0.0, device=device)
        loss_kd_step_r = torch.tensor(0.0, device=device)
        kd_active_ratio = 0.0
        if args.teacher == "nm" and args.teacher_prob > 0:
            with torch.no_grad():
                kd_mask = (snr_db >= float(args.teacher_snr_min)) & (
                    torch.rand(args.batch_size, device=device) < float(args.teacher_prob)
                )
                idx = torch.nonzero(kd_mask, as_tuple=False).squeeze(1)
                kd_active_ratio = float(idx.numel()) / float(args.batch_size)

            if idx.numel() > 0:
                z_np = z[idx].detach().cpu().numpy()
                th0_np = theta0[idx].detach().cpu().numpy().astype(np.float32)
                r0_np = r0[idx].detach().cpu().numpy().astype(np.float32)

                th_nm = np.zeros((idx.numel(),), dtype=np.float32)
                r_nm = np.zeros((idx.numel(),), dtype=np.float32)
                T = int(args.T)
                th_nm_hist = np.zeros((idx.numel(), T), dtype=np.float32) if need_kd_step else None
                r_nm_hist = np.zeros((idx.numel(), T), dtype=np.float32) if need_kd_step else None
                for j in range(idx.numel()):
                    maxiter = int(args.teacher_maxiter)
                    if need_kd_step:
                        maxiter = min(maxiter, T)
                        res, th_h, r_h = refine_nelder_mead_with_history(
                            z_np[j],
                            float(th0_np[j]),
                            float(r0_np[j]),
                            cfg,
                            theta_range=(box.theta_min, box.theta_max),
                            r_range=(box.r_min, box.r_max),
                            maxiter=maxiter,
                            hist_len=T,
                        )
                        th_nm_hist[j, :] = th_h
                        r_nm_hist[j, :] = r_h
                    else:
                        res = refine_nelder_mead(
                            z_np[j],
                            float(th0_np[j]),
                            float(r0_np[j]),
                            cfg,
                            theta_range=(box.theta_min, box.theta_max),
                            r_range=(box.r_min, box.r_max),
                            maxiter=maxiter,
                        )
                    th_nm[j] = res.theta_deg
                    r_nm[j] = res.r_m

                th_nm_t = torch.from_numpy(th_nm).to(device=device)
                r_nm_t = torch.from_numpy(r_nm).to(device=device)
                theta_nm_err = angle_error_deg_torch(theta_hat[idx], th_nm_t)
                r_nm_err = r_hat[idx] - r_nm_t
                loss_nm = mse_to_zero(theta_nm_err) + mse_to_zero(r_nm_err)
                snr_kd = snr_db[idx]
                kd_snr_ge0 = snr_kd >= 0.0
                loss_nm_snr_ge0 = masked_pair_mse(theta_nm_err, r_nm_err, kd_snr_ge0)
                loss_nm_snr_lt0 = masked_pair_mse(theta_nm_err, r_nm_err, ~kd_snr_ge0)

                if need_kd_step and trace is not None and float(args.w_kd_step) > 0.0:
                    if th_nm_hist is None or r_nm_hist is None:
                        raise RuntimeError("Missing NM history for step-wise KD")
                    w_t = t_weights(T, kd_time_weight, device)  # (T,)

                    # Teacher trajectory (T,Bkd)
                    th_teacher = torch.from_numpy(th_nm_hist).to(device=device).transpose(0, 1)
                    r_teacher = torch.from_numpy(r_nm_hist).to(device=device).transpose(0, 1)

                    # Student u trajectory: u_states_all[t] is u after t updates; u_states_all[T] is final.
                    u_pre = trace["u"]
                    if len(u_pre) < T:
                        raise RuntimeError("Unexpected u trace length in refiner")
                    u_states_all = torch.stack([*u_pre[:T], trace["u_T"]], dim=0)  # (T+1,B,2)

                    # Student absolute trajectory after each update (T,Bkd)
                    u_post = u_states_all[1:, idx, :].reshape(T * idx.numel(), 2)
                    th_s_flat, r_s_flat = map_u_to_theta_r(u_post, refiner.box, r_box=refiner.r_box)
                    th_student = th_s_flat.view(T, -1)
                    r_student = r_s_flat.view(T, -1)

                    # Normalized trajectory alignment (theta wrap + r scaled)
                    e_th = angle_error_deg_torch(th_student, th_teacher)  # (T,Bkd)
                    e_r = (r_student - r_teacher) / float(args.r_scale)  # (T,Bkd)
                    term_th = torch.mean((e_th / float(args.theta_scale)) ** 2, dim=1)  # (T,)
                    term_r = torch.mean(e_r**2, dim=1)  # (T,)
                    loss_kd_step_theta = (w_t * term_th).sum()
                    loss_kd_step_r = (w_t * term_r).sum()
                    kd_step = float(args.w_kd_step_theta) * loss_kd_step_theta + float(args.w_kd_step_r) * loss_kd_step_r
                    loss_kd_step = float(args.w_kd_step) * kd_step

        if dump_this_step and trace is not None:
            dump_n = min(2, int(args.batch_size))
            dump_idx = torch.randperm(int(args.batch_size), device=device)[:dump_n]
            z_dump = z[dump_idx].detach().cpu().numpy()
            th0_dump = theta0[dump_idx].detach().cpu().numpy().astype(np.float32)
            r0_dump = r0[dump_idx].detach().cpu().numpy().astype(np.float32)
            th_gt_dump = theta_gt[dump_idx].detach().cpu().numpy().astype(np.float32)
            r_gt_dump = r_gt[dump_idx].detach().cpu().numpy().astype(np.float32)
            snr_dump = snr_db[dump_idx].detach().cpu().numpy().astype(np.float32)

            T = int(args.T)
            th_nm_hist_dump = np.zeros((dump_n, T), dtype=np.float32)
            r_nm_hist_dump = np.zeros((dump_n, T), dtype=np.float32)
            for j in range(dump_n):
                maxiter = min(int(args.teacher_maxiter), T)
                _, th_h, r_h = refine_nelder_mead_with_history(
                    z_dump[j],
                    float(th0_dump[j]),
                    float(r0_dump[j]),
                    cfg,
                    theta_range=(box.theta_min, box.theta_max),
                    r_range=(box.r_min, box.r_max),
                    maxiter=maxiter,
                    hist_len=T,
                )
                th_nm_hist_dump[j, :] = th_h
                r_nm_hist_dump[j, :] = r_h

            u_pre = trace["u"]
            if len(u_pre) < T:
                raise RuntimeError("Unexpected u trace length in refiner")
            u_states_all = torch.stack([*u_pre[:T], trace["u_T"]], dim=0)  # (T+1,B,2)
            u_sel = u_states_all[:, dump_idx, :].reshape((T + 1) * dump_n, 2)
            th_s, r_s = map_u_to_theta_r(u_sel, refiner.box, r_box=refiner.r_box)
            th_pred_hist = th_s.view(T + 1, dump_n).detach().cpu().numpy().astype(np.float32)
            r_pred_hist = r_s.view(T + 1, dump_n).detach().cpu().numpy().astype(np.float32)

            trace_dir = run_dir / "traces"
            ensure_dir(trace_dir)
            out_path = trace_dir / f"step_{step}.npz"
            np.savez(
                out_path,
                theta0=th0_dump,
                r0=r0_dump,
                theta_gt=th_gt_dump,
                r_gt=r_gt_dump,
                theta_pred_hist=th_pred_hist,
                r_pred_hist=r_pred_hist,
                theta_nm_hist=np.concatenate([th0_dump[:, None], th_nm_hist_dump], axis=1),
                r_nm_hist=np.concatenate([r0_dump[:, None], r_nm_hist_dump], axis=1),
                snr_db=snr_dump,
            )

        loss_gt, loss_gt_theta, loss_gt_r = gt_loss(
            theta_err,
            r_err,
            theta_scale=float(args.theta_scale),
            r_scale=float(args.r_scale),
            w_theta=float(args.w_theta),
            w_r=float(args.w_r),
            huber_delta=float(args.huber_delta),
        )
        snr_ge0 = snr_db >= 0.0
        loss_gt_snr_ge0 = masked_gt_loss(
            theta_err,
            r_err,
            snr_ge0,
            theta_scale=float(args.theta_scale),
            r_scale=float(args.r_scale),
            w_theta=float(args.w_theta),
            w_r=float(args.w_r),
            huber_delta=float(args.huber_delta),
        )
        loss_gt_snr_lt0 = masked_gt_loss(
            theta_err,
            r_err,
            ~snr_ge0,
            theta_scale=float(args.theta_scale),
            r_scale=float(args.r_scale),
            w_theta=float(args.w_theta),
            w_r=float(args.w_r),
            huber_delta=float(args.huber_delta),
        )
        ll_mean = dbg["ll_mean"]
        loss_pscale_reg = (
            float(args.pscale_reg_w) * dbg["pscale_reg_term"]
            if "pscale_reg_term" in dbg and float(args.pscale_reg_w) > 0
            else torch.tensor(0.0, device=device)
        )
        loss = (
            float(args.w_nm) * loss_nm
            + float(args.w_gt) * loss_gt
            + float(nm_lambda_eff_used) * (-ll_mean)
            + loss_kd_step
            + loss_pscale_reg
        )

        if not torch.isfinite(loss):
            lr0 = float(opt.param_groups[0].get("lr", args.lr))
            print(
                "[warn] NaN/Inf loss; skip optimizer step "
                f"(step={step} nm_lambda_eff_raw={nm_lambda_eff_raw:.6g} nm_lambda_eff_used={nm_lambda_eff_used:.6g} lr={lr0:.6g})"
            )
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0.0:
            grad_norm_total = torch.nn.utils.clip_grad_norm_(refiner.parameters(), float(args.grad_clip))
        else:
            grad_norm_total = total_grad_norm_l2(refiner.parameters()).to(device=device)

        if not torch.isfinite(grad_norm_total):
            lr0 = float(opt.param_groups[0].get("lr", args.lr))
            print(
                "[warn] NaN/Inf grad norm; skip optimizer step "
                f"(step={step} nm_lambda_eff_raw={nm_lambda_eff_raw:.6g} nm_lambda_eff_used={nm_lambda_eff_used:.6g} lr={lr0:.6g})"
            )
            continue
        opt.step()

        with torch.no_grad():
            params_finite = all(torch.isfinite(p).all().item() for p in refiner.parameters())
            if not params_finite:
                print(f"NaN/Inf parameters encountered at step={step}. Rollback+save+stop.")
                refiner.load_state_dict(last_good_state, strict=True)
                break
            last_good_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
            last_good_step = step

        if step % args.log_interval == 0:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "rmse_theta_deg": float(torch.sqrt(torch.mean(theta_err * theta_err)).item()),
                "rmse_r_m": float(torch.sqrt(torch.mean(r_err * r_err)).item()),
                "ll_mean": float(ll_mean.item()),
                "alpha_mean": float(dbg["alpha_mean"].detach().cpu().item()),
                "lambda_mean": float(dbg["lambda_mean"].detach().cpu().item()),
                "beta": float(beta),
                "grad_norm": float(
                    grad_norm_total.item() if isinstance(grad_norm_total, torch.Tensor) else grad_norm_total
                ),
                "grad_clip": float(args.grad_clip),
                "loss_nm": float(loss_nm.item()),
                "loss_gt": float(loss_gt.item()),
                "loss_gt_theta": float(loss_gt_theta.item()),
                "loss_gt_r": float(loss_gt_r.item()),
                "kd_active_ratio": float(kd_active_ratio),
                "loss_gt_snr_ge0": float(loss_gt_snr_ge0.item()),
                "loss_gt_snr_lt0": float(loss_gt_snr_lt0.item()),
                "loss_nm_snr_ge0": float(loss_nm_snr_ge0.item()),
                "loss_nm_snr_lt0": float(loss_nm_snr_lt0.item()),
                "loss_kd_step": float(loss_kd_step.item()),
                "loss_kd_step_theta": float(loss_kd_step_theta.item()),
                "loss_kd_step_r": float(loss_kd_step_r.item()),
                "pscale_reg_loss": float(loss_pscale_reg.item()),
                "t_table_is_frozen": bool(t_table_is_frozen),
                "nm_lambda_eff": float(nm_lambda_eff_used),
                "nm_lambda_eff_raw": float(nm_lambda_eff_raw),
                "nm_lambda_eff_used": float(nm_lambda_eff_used),
            }
            if "pscale_scale_mean" in dbg:
                m = dbg["pscale_scale_mean"].detach().cpu().to(torch.float32)
                s = dbg["pscale_scale_std"].detach().cpu().to(torch.float32)
                row.update(
                    {
                        "pscale_mean_theta": float(m[0].item()),
                        "pscale_mean_r": float(m[1].item()),
                        "pscale_std_theta": float(s[0].item()),
                        "pscale_std_r": float(s[1].item()),
                        "pscale_clamp_hit_ratio_theta": float(dbg["pscale_clamp_hit_ratio"][0].detach().cpu().item()),
                        "pscale_clamp_hit_ratio_r": float(dbg["pscale_clamp_hit_ratio"][1].detach().cpu().item()),
                        "pscale_clamp_hit_ratio": float(dbg["pscale_clamp_hit_ratio"].mean().detach().cpu().item()),
                    }
                )
            if "t_table_mean" in dbg:
                tm = dbg["t_table_mean"].detach().cpu().to(torch.float32)
                ts = dbg["t_table_std"].detach().cpu().to(torch.float32)
                row.update(
                    {
                        "t_table_mean_theta": float(tm[0].item()),
                        "t_table_mean_r": float(tm[1].item()),
                        "t_table_std_theta": float(ts[0].item()),
                        "t_table_std_r": float(ts[1].item()),
                    }
                )
            logger.log(row)
            print(row)

        if step % args.val_interval == 0:
            refiner.eval()
            rmses_t = []
            rmses_r = []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    theta_gt, r_gt, snr_db = sample_gt(
                        args.batch_size, box, args.snr_min, args.snr_max, device
                    )
                    _, z, _ = synthesize_batch_torch(theta_gt, r_gt, snr_db, cfg)
                    theta0, r0, _ = coarse.search(z)
                    theta_hat, r_hat, _ = refiner(z, theta0, r0, beta=beta)
                    theta_err = angle_error_deg_torch(theta_hat, theta_gt)
                    r_err = r_hat - r_gt
                    rmses_t.append(torch.sqrt(torch.mean(theta_err * theta_err)).item())
                    rmses_r.append(torch.sqrt(torch.mean(r_err * r_err)).item())
            val_row = {
                "step": step,
                "val_rmse_theta_deg": float(np.mean(rmses_t)),
                "val_rmse_r_m": float(np.mean(rmses_r)),
            }
            val_score = float(val_row["val_rmse_theta_deg"] + val_row["val_rmse_r_m"] / float(R0))
            if args.early_stop_metric == "val_score":
                current_metric = val_score
            elif args.early_stop_metric == "theta":
                current_metric = float(val_row["val_rmse_theta_deg"])
            elif args.early_stop_metric == "r":
                current_metric = float(val_row["val_rmse_r_m"])
            else:
                raise RuntimeError(f"Unknown early_stop_metric: {args.early_stop_metric}")

            improved = (best_metric - current_metric) > float(args.early_stop_min_delta)
            if improved:
                best_metric = float(current_metric)
                bad_epochs = 0
                best_ckpt = {
                    "state_dict": refiner.state_dict(),
                    "T": int(args.T),
                    "cfg": cfg.__dict__,
                    "args": vars(args),
                    "best_metric": float(best_metric),
                    "best_step": int(step),
                }
                torch.save(best_ckpt, run_dir / "best.pt")
            else:
                bad_epochs += 1

            val_row = {
                **val_row,
                "val_score": float(val_score),
                "best_metric": float(best_metric),
                "bad_epochs": int(bad_epochs),
                "patience": int(args.early_stop_patience),
            }
            logger.log(val_row)
            print(
                f"[val] step={step} "
                f"rmse_theta_deg={val_row['val_rmse_theta_deg']:.6g} "
                f"rmse_r_m={val_row['val_rmse_r_m']:.6g} "
                f"val_score={val_row['val_score']:.6g} "
                f"best_metric={best_metric:.6g} "
                f"bad={bad_epochs}/{int(args.early_stop_patience)}"
            )
            if bad_epochs >= int(args.early_stop_patience):
                print(
                    f"[early-stop] step={step} metric={args.early_stop_metric} "
                    f"current={current_metric:.6g} best={best_metric:.6g} "
                    f"min_delta={float(args.early_stop_min_delta):.6g} "
                    f"patience={int(args.early_stop_patience)}"
                )
                break
            refiner.train()

    state_to_save = (
        refiner.state_dict()
        if all(torch.isfinite(p).all().item() for p in refiner.parameters())
        else last_good_state
    )
    ckpt = {
        "state_dict": state_to_save,
        "T": int(args.T),
        "cfg": cfg.__dict__,
        "args": vars(args),
        "last_good_step": int(last_good_step),
    }
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
