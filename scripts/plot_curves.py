from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="runs/latest_eval")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()


def read_results_csv(path: Path):
    import csv

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "results.csv").exists():
        # If symlink is unavailable on Windows, fall back to a pointer file created by eval_all.
        ptr = run_dir.parent / f"{run_dir.name}_path.txt"
        if ptr.exists():
            target = ptr.read_text(encoding="utf-8").strip()
            if target:
                run_dir = Path(target)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results_csv(run_dir / "results.csv")
    methods = sorted({r["method"] for r in rows})
    snrs = sorted({float(r["snr_db"]) for r in rows})

    def get_curve(method: str, key: str):
        y = []
        for snr in snrs:
            rr = [r for r in rows if r["method"] == method and float(r["snr_db"]) == snr]
            if not rr or rr[0][key] in ("", None):
                y.append(np.nan)
            else:
                y.append(float(rr[0][key]))
        return np.array(y, dtype=np.float64)

    plt.figure(figsize=(7, 4))
    for m in methods:
        y = get_curve(m, "rmse_theta_deg")
        plt.plot(snrs, y, marker="o", label=m)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE θ (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_theta.png", dpi=150)

    plt.figure(figsize=(7, 4))
    for m in methods:
        y = get_curve(m, "rmse_r_m")
        plt.plot(snrs, y, marker="o", label=m)
    plt.xlabel("SNR (dB)")
    plt.ylabel("RMSE r (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_r.png", dpi=150)

    plt.figure(figsize=(7, 4))
    for m in methods:
        y = get_curve(m, "ms_per_sample")
        plt.plot(snrs, y, marker="o", label=m)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Time (ms/sample)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_ms.png", dpi=150)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
