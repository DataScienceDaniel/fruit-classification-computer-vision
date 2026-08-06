"""Benchmark de latência de inferência.

Mede em CPU além de GPU: um quiosque de pesagem raramente traz GPU discreta,
então o número de CPU é o relevante para sustentar o argumento de edge.
Inclui warm-up (a primeira inferência carrega o CUDA e distorce a média) e
sincronização explícita, sem a qual as medições em GPU ficam otimistas.
"""

from __future__ import annotations

import time

import numpy as np


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _sync(device: str) -> None:
    if device == "cuda" and _cuda_available():
        import torch

        torch.cuda.synchronize()


def benchmark_latency(
    ckpt: str,
    sample_path: str,
    device: str,
    runs: int = 50,
    warmup: int = 10,
    imgsz: int = 224,
) -> dict:
    """Latência de uma imagem, em milissegundos."""
    from ultralytics import YOLO

    model = YOLO(ckpt)

    for _ in range(warmup):
        model.predict(sample_path, imgsz=imgsz, verbose=False, device=device)
    _sync(device)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(sample_path, imgsz=imgsz, verbose=False, device=device)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "device": device,
        "runs": runs,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def benchmark_all_devices(ckpt: str, sample_path: str, imgsz: int = 224) -> list[dict]:
    """Mede em GPU (se houver) e em CPU, nessa ordem."""
    stats = []
    if _cuda_available():
        stats.append(benchmark_latency(ckpt, sample_path, "cuda", imgsz=imgsz))
    stats.append(benchmark_latency(ckpt, sample_path, "cpu", imgsz=imgsz))
    return stats
