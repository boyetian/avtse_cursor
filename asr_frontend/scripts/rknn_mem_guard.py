"""Abort RKNN conversion when free RAM is below threshold (no swap)."""

from __future__ import annotations

import sys


def avail_mem_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def require_avail_gb(min_gb: float, label: str = "RKNN conversion") -> None:
    avail = avail_mem_gb()
    if avail < min_gb:
        print(
            f"skip {label}: MemAvailable={avail:.1f}GiB < required {min_gb:.1f}GiB "
            "(not enough RAM; will not use swap)",
            file=sys.stderr,
        )
        raise SystemExit(2)
