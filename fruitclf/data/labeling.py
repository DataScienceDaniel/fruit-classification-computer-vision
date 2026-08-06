"""Extração de rótulos a partir de nomes de arquivo.

Duas taxonomias são derivadas aqui:
  label_8  — apenas o tipo de produto;
  label_14 — produto × condição de embalagem (with_bag / without_bag).

O atributo de embalagem é heurístico e deve ser auditado (ver
:mod:`fruitclf.data.audit`) antes de ser reportado como rótulo confiável.
"""

from __future__ import annotations

import re

import pandas as pd


def extract_label(name: str) -> str | None:
    """Extrai o rótulo do produto a partir do nome do arquivo."""
    m = re.search(r"^\d+_\d+_([a-zA-Z]+)_", name)
    if m:
        return m.group(1).lower()
    m = re.search(r"^([a-zA-Z]+(?:\s[a-zA-Z]+)?)\s*-\s*\d+", name)
    if m:
        return m.group(1).strip().lower()
    m = re.search(r"([a-zA-Z]+)", name)
    if m:
        return m.group(1).lower()
    return None


def extract_bag(name: str) -> str:
    """Detecta a condição de embalagem: 'Yes', 'No' ou 'Unknown'.

    A ordem de teste importa: o padrão sem-sacola é verificado primeiro, e
    ambos usam limites de token para cobrir variações como ``wo_b`` ou ``w-b``,
    que uma checagem por substring simples classificaria errado.
    """
    low = name.lower()
    if re.search(r"(?:^|[_\-\s])w[_\-\s]?o[_\-\s]?b(?:$|[_\-\s\d.])", low):
        return "No"
    if re.search(r"(?:^|[_\-\s])w[_\-\s]?b(?:$|[_\-\s\d.])", low):
        return "Yes"
    return "Unknown"


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as colunas Label, Bag, label_8 e label_14."""
    df = df.copy()
    df["Label"] = df["filename"].apply(extract_label)
    df["Bag"] = df["filename"].apply(extract_bag)
    df = df[df["Label"].notna()].reset_index(drop=True)

    df["label_8"] = df["Label"].str.replace(" ", "_", regex=False)

    def fine(row: pd.Series) -> str:
        base = row["label_8"]
        if row["Bag"] == "Yes":
            return f"{base}_with_bag"
        if row["Bag"] == "No":
            return f"{base}_without_bag"
        return base

    df["label_14"] = df.apply(fine, axis=1)
    return df


def describe(df: pd.DataFrame) -> str:
    """Resumo textual da composição do dataset, para log e para o artigo."""
    parts = [
        f"total de imagens: {len(df)}",
        "",
        "label_8:",
        df["label_8"].value_counts().sort_index().to_string(),
        "",
        "label_14:",
        df["label_14"].value_counts().sort_index().to_string(),
        "",
        "Bag:",
        df["Bag"].value_counts().to_string(),
    ]
    return "\n".join(parts)
