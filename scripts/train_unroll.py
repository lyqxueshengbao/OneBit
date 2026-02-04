from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.coarse_search import CoarseSearcherTorch
from src.dataset import TargetBox, synthesize_batch_torch
from src.fda import FDAConfig
from src.metrics import angle_error_deg_torch
from src.nm_refine import refine_nelder_mead_with_history
from src.unroll_refine import Box, Refiner
from src.utils import JsonlLogger, ensure_dir, seed_all, timestamp, write_json


R0 = 50.0
HUBER_DELTA_THETA_DEG = 1.0
HUBER_DELTA_R_M = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
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
    p.add_argument("--w_gt", type=float, default=0.2)
    p.add_argument("--w_phys", type=float, default=0.05)
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def mse_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x * x)


def masked_mse(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("mask must be a bool tensor")
    if mask.numel() == 0 or not mask.any().item():
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return mse_to_zero(x[mask])


def masked_pair_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mse(x, mask) + masked_mse(y, mask)


def t_weights(T: int, mode: str, device: torch.device) -> torch.Tensor:
    if T <= 0:
        raise ValueError("T must be positive")
    if mode == "uniform":
        w = torch.ones((T,), device=device, dtype=torch.float32)
    elif mode == "late":
        t = torch.arange(1, T + 1, device=device, dtype=torch.float32) / float(T)
        w = t * t
    else:
        raise ValueError(f"Unknown kd_step_t_weight: {mode}")
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
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = FDAConfig()
    box = TargetBox()

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / f"train_{timestamp()}"
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
    ).to(device)

    opt = torch.optim.Adam(refiner.parameters(), lr=args.lr)
    last_good_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
    last_good_step = 0
    best_metric = float("inf")
    bad_epochs = 0

    for step in range(1, args.steps + 1):
        beta = beta_schedule(step, args.steps, args.beta, args.beta_final, args.beta_warmup_frac)
        theta_gt, r_gt, snr_db = sample_gt(
            args.batch_size, box, args.snr_min, args.snr_max, device
        )
        _, z, _ = synthesize_batch_torch(theta_gt, r_gt, snr_db, cfg)

        theta0, r0, _ = coarse.search(z)
        want_trace = (
            args.teacher == "nm"
            and float(args.teacher_prob) > 0.0
            and float(args.w_kd_step) > 0.0
            and (float(args.w_kd_step_theta) > 0.0 or float(args.w_kd_step_r) > 0.0)
        )
        if want_trace:
            theta_hat, r_hat, dbg, trace = refiner(z, theta0, r0, beta=beta, return_trace=True)
        else:
            theta_hat, r_hat, dbg = refiner(z, theta0, r0, beta=beta)
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
                th_nm_hist = np.zeros((idx.numel(), int(args.T)), dtype=np.float32)
                r_nm_hist = np.zeros((idx.numel(), int(args.T)), dtype=np.float32)
                for j in range(idx.numel()):
                    maxiter = min(int(args.teacher_maxiter), int(args.T))
                    res, th_h, r_h = refine_nelder_mead_with_history(
                        z_np[j],
                        float(th0_np[j]),
                        float(r0_np[j]),
                        cfg,
                        theta_range=(box.theta_min, box.theta_max),
                        r_range=(box.r_min, box.r_max),
                        maxiter=maxiter,
                        hist_len=int(args.T),
                    )
                    th_nm[j] = res.theta_deg
                    r_nm[j] = res.r_m
                    th_nm_hist[j, :] = th_h
                    r_nm_hist[j, :] = r_h

                th_nm_t = torch.from_numpy(th_nm).to(device=device)
                r_nm_t = torch.from_numpy(r_nm).to(device=device)
                theta_nm_err = angle_error_deg_torch(theta_hat[idx], th_nm_t)
                r_nm_err = r_hat[idx] - r_nm_t
                loss_nm = mse_to_zero(theta_nm_err) + mse_to_zero(r_nm_err)
                snr_kd = snr_db[idx]
                kd_snr_ge0 = snr_kd >= 0.0
                loss_nm_snr_ge0 = masked_pair_mse(theta_nm_err, r_nm_err, kd_snr_ge0)
                loss_nm_snr_lt0 = masked_pair_mse(theta_nm_err, r_nm_err, ~kd_snr_ge0)

                if want_trace and trace is not None and float(args.w_kd_step) > 0.0:
                    # Student per-step states AFTER each update:
                    # trace["theta"][t] is pre-update state; post-update for step t is next pre-update (or final).
                    theta_pre = trace["theta"]
                    r_pre = trace["r"]
                    if len(theta_pre) < int(args.T) or len(r_pre) < int(args.T):
                        raise RuntimeError("Unexpected trace length in refiner")
                    theta_steps = torch.stack([*theta_pre[1:int(args.T)], theta_hat.detach()], dim=0)  # (T,B)
                    r_steps = torch.stack([*r_pre[1:int(args.T)], r_hat.detach()], dim=0)  # (T,B)

                    th_teacher = torch.from_numpy(th_nm_hist).to(device=device).transpose(0, 1)  # (T,Bkd)
                    r_teacher = torch.from_numpy(r_nm_hist).to(device=device).transpose(0, 1)  # (T,Bkd)
                    th_student = theta_steps[:, idx]
                    r_student = r_steps[:, idx]

                    w_t = t_weights(int(args.T), str(args.kd_step_t_weight), device).view(-1, 1)  # (T,1)
                    hub_th = torch.nn.functional.huber_loss(
                        th_student.to(torch.float32),
                        th_teacher,
                        delta=float(HUBER_DELTA_THETA_DEG),
                        reduction="none",
                    )
                    hub_r = torch.nn.functional.huber_loss(
                        r_student.to(torch.float32),
                        r_teacher,
                        delta=float(HUBER_DELTA_R_M),
                        reduction="none",
                    )
                    loss_kd_step_theta = (w_t * hub_th.mean(dim=1, keepdim=True)).sum()
                    loss_kd_step_r = (w_t * hub_r.mean(dim=1, keepdim=True)).sum()
                    kd_step = float(args.w_kd_step_theta) * loss_kd_step_theta + float(args.w_kd_step_r) * loss_kd_step_r
                    loss_kd_step = float(args.w_kd_step) * kd_step

        loss_gt = mse_to_zero(theta_err) + mse_to_zero(r_err)
        snr_ge0 = snr_db >= 0.0
        loss_gt_snr_ge0 = masked_pair_mse(theta_err, r_err, snr_ge0)
        loss_gt_snr_lt0 = masked_pair_mse(theta_err, r_err, ~snr_ge0)
        ll_mean = dbg["ll_mean"]
        loss = (
            float(args.w_nm) * loss_nm
            + float(args.w_gt) * loss_gt
            + float(args.w_phys) * (-ll_mean)
            + loss_kd_step
        )

        if not torch.isfinite(loss):
            print("NaN/Inf loss encountered. Rollback+save+stop.")
            refiner.load_state_dict(last_good_state, strict=True)
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), float(args.grad_clip))
        if not torch.isfinite(grad_norm):
            print("NaN/Inf grad norm encountered. Rollback+save+stop.")
            refiner.load_state_dict(last_good_state, strict=True)
            break
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
                "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "loss_nm": float(loss_nm.item()),
                "loss_gt": float(loss_gt.item()),
                "kd_active_ratio": float(kd_active_ratio),
                "loss_gt_snr_ge0": float(loss_gt_snr_ge0.item()),
                "loss_gt_snr_lt0": float(loss_gt_snr_lt0.item()),
                "loss_nm_snr_ge0": float(loss_nm_snr_ge0.item()),
                "loss_nm_snr_lt0": float(loss_nm_snr_lt0.item()),
                "loss_kd_step": float(loss_kd_step.item()),
                "loss_kd_step_theta": float(loss_kd_step_theta.item()),
                "loss_kd_step_r": float(loss_kd_step_r.item()),
            }
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
