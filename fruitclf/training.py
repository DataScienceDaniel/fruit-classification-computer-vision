"""Fine-tuning do classificador YOLOv8s.

Todas as configurações usam os mesmos hiperparâmetros e a mesma semente, de
modo que as diferenças entre elas sejam atribuíveis apenas ao split.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from fruitclf.config import PRETRAINED, TrainConfig


def train_classifier(dataset_dir: Path, out_dir: Path, cfg: TrainConfig):
    """Treina e devolve (caminho do checkpoint, diretório do run)."""
    from ultralytics import YOLO

    model = YOLO(PRETRAINED)
    model.train(
        data=str(dataset_dir),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        project=str(out_dir / "runs"),
        name="train",
        exist_ok=True,
    )

    runs = sorted(glob.glob(str(out_dir / "runs" / "*")), key=os.path.getmtime)
    if not runs:
        raise RuntimeError(f"nenhum run encontrado em {out_dir / 'runs'}")
    run_dir = runs[-1]
    ckpt = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"checkpoint nao encontrado: {ckpt}")
    return ckpt, run_dir


def load_checkpoint(ckpt: str):
    from ultralytics import YOLO

    return YOLO(ckpt)


def model_size_mb(ckpt: str) -> float:
    return os.path.getsize(ckpt) / (1024**2)
