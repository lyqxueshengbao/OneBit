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
from src.unroll_refine import Box, Refiner
from src.utils import JsonlLogger, ensure_dir, seed_all, timestamp, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--snr_min", type=float, default=-15.0)
    p.add_argument("--snr_max", type=float, default=15.0)
    p.add_argument("--theta_step", type=float, default=1.0)
    p.add_argument("--r_step", type=float, default=100.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=200)
    p.add_argument("--val_batches", type=int, default=5)
    p.add_argument("--physics_weight", type=float, default=0.0)  # set 0.1 to enable
    p.add_argument("--run_dir", type=str, default="")
    return p.parse_args()


def smooth_l1_to_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(x, torch.zeros_like(x))


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

    for step in range(1, args.steps + 1):
        theta_gt, r_gt, snr_db = sample_gt(
            args.batch_size, box, args.snr_min, args.snr_max, device
        )
        _, z, _ = synthesize_batch_torch(theta_gt, r_gt, snr_db, cfg)

        theta0, r0, _ = coarse.search(z)
        theta_hat, r_hat, J_hat = refiner(z, theta0, r0)

        theta_err = angle_error_deg_torch(theta_hat, theta_gt)
        r_err = r_hat - r_gt

        loss = smooth_l1_to_zero(theta_err) + smooth_l1_to_zero(r_err)
        if args.physics_weight > 0:
            loss = loss + float(args.physics_weight) * torch.mean(-J_hat)

        if not torch.isfinite(loss):
            print("Loss is NaN/Inf, stopping.")
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        opt.step()

        if step % args.log_interval == 0:
            row = {
                "step": step,
                "loss": float(loss.item()),
                "rmse_theta_deg": float(torch.sqrt(torch.mean(theta_err * theta_err)).item()),
                "rmse_r_m": float(torch.sqrt(torch.mean(r_err * r_err)).item()),
                "alpha_mean": float(refiner.alpha().mean().item()),
                "lambda_mean": float(refiner.lambd().mean().item()),
                "J_mean": float(J_hat.mean().item()),
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
                    theta_hat, r_hat, _ = refiner(z, theta0, r0)
                    theta_err = angle_error_deg_torch(theta_hat, theta_gt)
                    r_err = r_hat - r_gt
                    rmses_t.append(torch.sqrt(torch.mean(theta_err * theta_err)).item())
                    rmses_r.append(torch.sqrt(torch.mean(r_err * r_err)).item())
            val_row = {
                "step": step,
                "val_rmse_theta_deg": float(np.mean(rmses_t)),
                "val_rmse_r_m": float(np.mean(rmses_r)),
            }
            logger.log(val_row)
            print(val_row)
            refiner.train()

    ckpt = {"state_dict": refiner.state_dict(), "cfg": cfg.__dict__, "args": vars(args)}
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
