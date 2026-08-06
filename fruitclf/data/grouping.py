"""Detecção de quase-duplicatas e formação de grupos.

O dataset de origem foi montado para detecção e contém sequências longas de
frames do mesmo item físico. Dividir no nível da imagem coloca frames quase
idênticos dos dois lados da partição, e o classificador atinge acurácia
perfeita memorizando itens. Este módulo agrupa esses frames para que o split
possa tratá-los como uma unidade indivisível.

Dois critérios, unidos por componentes conexas:
  (a) distância de Hamming entre dHashes dentro da mesma classe;
  (b) mesmo prefixo de nome de arquivo, quando o prefixo é discriminativo.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fruitclf.config import GroupingConfig


def dhash(path: str, hash_size: int = 8) -> np.ndarray | None:
    """Difference hash: 64 bits que capturam a estrutura da imagem."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(
        img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)


def name_prefix(fn: str) -> str:
    """Prefixo do nome do arquivo após remover UM contador final.

    Remover contadores repetidamente colapsa itens distintos: ``prod0_1`` e
    ``prod3_1`` virariam ambos ``prod``. Um único passe, portanto.
    """
    s = Path(fn).stem.lower()
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s)          # sufixo de export Roboflow
    s = re.sub(r"[_\-\s]?(jpg|jpeg|png)$", "", s)   # artefato '_jpg'
    s = re.sub(r"[_\-\s]?\d+$", "", s)              # UM contador final
    return s.strip("_- ") or Path(fn).stem.lower()


def assign_groups(
    df: pd.DataFrame,
    label_col: str,
    cfg: GroupingConfig | None = None,
) -> pd.DataFrame:
    """Adiciona a coluna ``group``: a unidade indivisível do split.

    Todas as imagens de um grupo vão inteiras para treino ou inteiras para
    validação.
    """
    cfg = cfg or GroupingConfig()
    df = df.copy().reset_index(drop=True)

    print("[grupos] calculando perceptual hashes...")
    hashes = [dhash(p) for p in df["path"]]
    valid = np.array([h is not None for h in hashes])
    if not valid.all():
        print(f"[grupos] {(~valid).sum()} imagens ilegiveis descartadas")
        df = df[valid].reset_index(drop=True)
        hashes = [h for h in hashes if h is not None]
    H = np.stack(hashes).astype(np.uint8)  # (n, 64)

    df["_prefix"] = df["filename"].apply(name_prefix)

    rows: list[int] = []
    cols: list[int] = []
    n = len(df)
    skipped_prefixes = 0

    for _, idx in df.groupby(label_col).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue

        # (a) quase-duplicatas visuais
        sub = H[idx]
        dist = (sub[:, None, :] != sub[None, :, :]).sum(axis=2)
        ii, jj = np.where(dist <= cfg.ham_thresh)
        rows.extend(idx[ii].tolist())
        cols.extend(idx[jj].tolist())

        # (b) mesmo prefixo, apenas se discriminativo
        if cfg.use_prefix:
            pref = df.loc[idx, "_prefix"].to_numpy()
            counts = Counter(pref)
            limit = max(2, int(cfg.prefix_max_frac * len(idx)))
            keep = np.array([counts[p] <= limit for p in pref])
            if keep.any():
                sel = np.where(keep)[0]
                p_sel = pref[sel]
                same = p_sel[:, None] == p_sel[None, :]
                ii, jj = np.where(same)
                rows.extend(idx[sel[ii]].tolist())
                cols.extend(idx[sel[jj]].tolist())
            skipped_prefixes += int((~keep).sum())

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    df["group"] = labels
    df = df.drop(columns=["_prefix"])

    sizes = Counter(labels)
    print(
        f"[grupos] {n} imagens -> {n_comp} grupos "
        f"(maior grupo: {max(sizes.values())}, "
        f"grupos unitarios: {sum(1 for v in sizes.values() if v == 1)})"
    )
    if skipped_prefixes:
        print(
            f"[grupos] {skipped_prefixes} imagens com prefixo nao discriminativo "
            f"(agrupadas apenas por similaridade visual)"
        )

    _warn_degenerate(df, label_col)
    return df


def _warn_degenerate(df: pd.DataFrame, label_col: str, min_groups: int = 3) -> None:
    """Avisa se alguma classe virou poucos grupos — o split ficaria frágil."""
    per_class = df.groupby(label_col)["group"].nunique()
    degenerate = per_class[per_class < min_groups]
    if len(degenerate):
        print(f"[AVISO] classes com menos de {min_groups} grupos:")
        print(degenerate.to_string())
        print("        considere ajustar ham_thresh ou desativar use_prefix")


def inspect_groups(
    df: pd.DataFrame,
    out_png: Path,
    n_groups: int = 6,
    n_per: int = 6,
) -> None:
    """Exporta um grid com amostras dos maiores grupos.

    Confira antes de treinar: cada linha deve conter frames do mesmo item
    físico. Linhas misturando itens distintos pedem ``ham_thresh`` menor;
    itens obviamente iguais em linhas diferentes pedem ``ham_thresh`` maior.
    """
    big = df["group"].value_counts().head(n_groups).index
    fig, axes = plt.subplots(len(big), n_per, figsize=(2 * n_per, 2 * len(big)))
    axes = np.atleast_2d(axes)
    for r, g in enumerate(big):
        sub = df[df["group"] == g].head(n_per)
        for c in range(n_per):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(sub):
                im = cv2.imread(sub.iloc[c]["path"])
                if im is not None:
                    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if c == 0:
                total = int((df["group"] == g).sum())
                ax.set_title(f"group {g} (n={total})", fontsize=8, loc="left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[grupos] preview salvo em {out_png}")
