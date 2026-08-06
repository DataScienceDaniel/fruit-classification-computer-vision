"""Constantes e configuração compartilhada."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 42
OUT_ROOT = Path("outputs")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KAGGLE_DATASET = "kvnpatel/fruits-vegetable-detection-for-yolov4"
KAGGLE_SUBDIRS = ("obj (1)", "test")

PRETRAINED = "yolov8s-cls.pt"


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes para que os experimentos sejam comparáveis entre si."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch só é necessário no treino/inferência
        pass


@dataclass
class TrainConfig:
    epochs: int = 50
    imgsz: int = 224
    batch: int = 16
    seed: int = SEED


@dataclass
class GroupingConfig:
    """Parâmetros da detecção de quase-duplicatas.

    ham_thresh: distância de Hamming máxima (0-64) entre dHashes.
    use_prefix: também agrupar por prefixo de nome de arquivo.
    prefix_max_frac: um prefixo que cobre mais que isso da classe identifica a
        classe, não um item, e é descartado.
    """

    ham_thresh: int = 8
    use_prefix: bool = True
    prefix_max_frac: float = 0.30


@dataclass
class ExperimentConfig:
    tag: str                      # "8" ou "14"
    label_col: str                # "label_8" ou "label_14"
    split_mode: str               # "random" ou "grouped"
    train: TrainConfig = field(default_factory=TrainConfig)
    robustness_subset: int | None = None

    @property
    def name(self) -> str:
        return f"{self.tag}_{self.split_mode}"

    @property
    def out_dir(self) -> Path:
        return OUT_ROOT / self.name

    @property
    def dataset_dir(self) -> Path:
        return Path(f"dataset_{self.name}")


TAXONOMIES = {"8": "label_8", "14": "label_14"}
