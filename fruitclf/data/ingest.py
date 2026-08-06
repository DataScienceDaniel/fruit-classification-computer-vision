"""Download do dataset e construção do DataFrame de imagens."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from fruitclf.config import IMG_EXTS, KAGGLE_DATASET, KAGGLE_SUBDIRS


def download_dataset() -> str:
    """Baixa o dataset do Kaggle e devolve o caminho local."""
    import kagglehub

    return kagglehub.dataset_download(KAGGLE_DATASET)


def build_image_df(base_dir: str) -> pd.DataFrame:
    """Varre um diretório de classes e devolve um DataFrame de imagens."""
    data = []
    for class_name in sorted(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if Path(fname).suffix.lower() in IMG_EXTS:
                data.append(
                    {
                        "source_dir": os.path.basename(base_dir),
                        "filename": fname,
                        "path": os.path.join(class_path, fname),
                    }
                )
    return pd.DataFrame(data)


def load_raw(root: str | None = None) -> pd.DataFrame:
    """Concatena todos os subdiretórios do dataset num único DataFrame."""
    root = root or download_dataset()
    frames = [build_image_df(os.path.join(root, sub)) for sub in KAGGLE_SUBDIRS]
    return pd.concat(frames, ignore_index=True)
