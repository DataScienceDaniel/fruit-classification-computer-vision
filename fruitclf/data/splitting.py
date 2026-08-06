"""Estratégias de partição treino/validação.

``split_random`` é mantido apenas como baseline com vazamento, para que a
diferença de acurácia entre as duas estratégias possa ser reportada.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from fruitclf.config import SEED


class LeakageError(AssertionError):
    """Levantado quando um grupo aparece nos dois lados da partição."""


class DegenerateSplitError(ValueError):
    """Levantado quando a partição é inutilizável (lado vazio ou classe ausente).

    Com poucos grupos, o ``StratifiedGroupKFold`` pode alocar tudo para um só
    lado sem levantar erro. Um treino vazio falharia muito depois, com uma
    mensagem que não aponta para a causa.
    """


def split_random(
    df: pd.DataFrame, label_col: str, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aleatório no nível da imagem — baseline com vazamento."""
    tr, va = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=SEED
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def split_grouped(
    df: pd.DataFrame, label_col: str, n_splits: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por grupo, estratificado por classe.

    ``n_splits=3`` produz cerca de 1/3 de validação. Reporte a proporção real
    devolvida por :func:`split_summary`, não o valor nominal — grupos têm
    tamanhos desiguais e o resultado não fecha exatamente em 70/30.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(sgkf.split(df, df[label_col], groups=df["group"]))
    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)

    overlap = set(tr["group"]) & set(va["group"])
    if overlap:
        raise LeakageError(f"{len(overlap)} grupos presentes em ambos os splits")

    _validate_split(df, tr, va, label_col)

    print(
        f"[split] treino={len(tr)} ({len(tr) / len(df):.1%}) | "
        f"val={len(va)} ({len(va) / len(df):.1%}) | grupos disjuntos: OK"
    )
    return tr, va


def _validate_split(
    df: pd.DataFrame, tr: pd.DataFrame, va: pd.DataFrame, label_col: str
) -> None:
    """Rejeita partições inutilizáveis antes que o treino comece."""
    if tr.empty or va.empty:
        raise DegenerateSplitError(
            f"split degenerado: treino={len(tr)}, val={len(va)}. "
            f"Grupos demais foram fundidos ({df['group'].nunique()} grupos para "
            f"{df[label_col].nunique()} classes) — reduza ham_thresh."
        )

    missing_tr = set(df[label_col]) - set(tr[label_col])
    missing_va = set(df[label_col]) - set(va[label_col])
    if missing_tr or missing_va:
        raise DegenerateSplitError(
            f"classes ausentes no treino: {sorted(missing_tr)}; "
            f"na validacao: {sorted(missing_va)}. "
            "Essas classes tem grupos demais fundidos ou amostras de menos."
        )


def split_summary(
    df: pd.DataFrame,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    label_col: str,
    split_mode: str,
) -> dict:
    """Contagens reais do split, para irem direto ao artigo."""
    info = {
        "split_mode": split_mode,
        "label_col": label_col,
        "n_total": int(len(df)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "train_frac": len(tr) / len(df),
        "val_frac": len(va) / len(df),
        "n_classes": int(df[label_col].nunique()),
        "class_counts": {
            k: int(v) for k, v in df[label_col].value_counts().sort_index().items()
        },
    }
    if "group" in df.columns:
        info["n_groups"] = int(df["group"].nunique())
    return info


def materialize(
    tr: pd.DataFrame, va: pd.DataFrame, label_col: str, dest: Path
) -> Path:
    """Copia as imagens para o layout de classificação do YOLO."""
    if dest.exists():
        shutil.rmtree(dest)
    for subset, sub_df in (("train", tr), ("val", va)):
        for _, row in sub_df.iterrows():
            folder = dest / subset / row[label_col]
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(row["path"], folder / os.path.basename(row["path"]))
    return dest
