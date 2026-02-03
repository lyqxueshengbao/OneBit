from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        ensure_dir(Path(self.path).parent)

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class CsvLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = str(path)
        self.fieldnames = fieldnames
        ensure_dir(Path(self.path).parent)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


@dataclass
class Timer:
    def __post_init__(self) -> None:
        self.t0 = 0.0
        self.dt = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
        return False


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

