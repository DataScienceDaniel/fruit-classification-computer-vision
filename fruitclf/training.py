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
        data=str(dataset_dir.resolve()),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        project=str((out_dir / "runs").resolve()),
        name="train",
        exist_ok=True,
    )

    # O Ultralytics reancora caminhos relativos no próprio runs/classify,
    # então o save_dir do trainer é a única fonte confiável.
    run_dir = Path(model.trainer.save_dir)
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint nao encontrado: {ckpt}")
    return str(ckpt), str(run_dir)


def load_checkpoint(ckpt: str):
    from ultralytics import YOLO

    return YOLO(ckpt)


def model_size_mb(ckpt: str) -> float:
    return os.path.getsize(ckpt) / (1024**2)
