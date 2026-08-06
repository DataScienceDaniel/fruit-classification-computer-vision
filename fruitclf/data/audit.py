"""Auditoria manual do atributo de embalagem.

Exporta uma amostra estratificada para conferência visual. Preencha a coluna
``bag_verdadeiro`` na planilha e rode :func:`agreement_rate` para obter a taxa
de concordância — isso transforma "possível ruído de rótulo" num número
reportável.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from fruitclf.config import SEED


def export_bag_audit(df: pd.DataFrame, out_dir: Path, n: int = 120) -> Path:
    """Copia uma amostra estratificada por condição de embalagem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_stratum = max(1, n // max(1, df["Bag"].nunique()))
    sample = df.groupby("Bag", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_stratum), random_state=SEED)
    )

    recs = []
    for i, row in enumerate(sample.itertuples()):
        dst = out_dir / f"{i:03d}_{row.Bag}_{Path(row.path).name}"
        shutil.copy(row.path, dst)
        recs.append(
            {
                "file": dst.name,
                "bag_heuristico": row.Bag,
                "label": row.label_8,
                "bag_verdadeiro": "",
            }
        )

    sheet = out_dir / "audit_sheet.csv"
    pd.DataFrame(recs).to_csv(sheet, index=False)
    print(f"[auditoria] {len(recs)} imagens em {out_dir}")
    print(f"[auditoria] preencha 'bag_verdadeiro' em {sheet}")
    return sheet


def agreement_rate(sheet: Path) -> dict:
    """Calcula a concordância entre a heurística e a anotação visual."""
    df = pd.read_csv(sheet)
    done = df[df["bag_verdadeiro"].notna() & (df["bag_verdadeiro"] != "")]
    if done.empty:
        raise ValueError("nenhuma linha preenchida em 'bag_verdadeiro'")

    match = done["bag_heuristico"].str.strip().str.lower() == done[
        "bag_verdadeiro"
    ].astype(str).str.strip().str.lower()

    disagreements = done[~match][["file", "bag_heuristico", "bag_verdadeiro"]]
    result = {
        "n_audited": int(len(done)),
        "n_agree": int(match.sum()),
        "agreement": float(match.mean()),
        "disagreements": disagreements.to_dict("records"),
    }
    print(
        f"[auditoria] concordancia: {result['agreement']:.1%} "
        f"({result['n_agree']}/{result['n_audited']})"
    )
    return result
