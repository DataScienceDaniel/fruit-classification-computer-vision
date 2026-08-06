"""Robustez sob perturbações controladas.

Aplicadas a todas as imagens de validação, não a um exemplo ilustrativo: a
ordenação das perturbações por impacto indica quais condições de aquisição a
câmera do quiosque precisa controlar.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from fruitclf.evaluation.reporting import robustness_latex

PERTURBATIONS = ("original", "bright", "dark", "occlusion", "blur", "jpeg")


def perturb(img: np.ndarray, kind: str) -> np.ndarray:
    h, w = img.shape[:2]
    if kind == "original":
        return img
    if kind == "bright":
        return cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    if kind == "dark":
        return cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
    if kind == "occlusion":
        out = img.copy()
        out[h // 4 : h // 2, w // 4 : w // 2] = 0
        return out
    if kind == "blur":
        k = max(3, (min(h, w) // 60) * 2 + 1)
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    raise ValueError(f"perturbacao desconhecida: {kind}")


def robustness_sweep(
    model,
    val_items: list[tuple[str, str]],
    out_dir: Path,
    tag: str,
    imgsz: int = 224,
    max_images: int | None = None,
    batch: int = 32,
) -> dict:
    """Acurácia sob cada perturbação."""
    from fruitclf.evaluation.metrics import class_index

    items = (
        val_items
        if max_images is None
        else random.sample(val_items, min(max_images, len(val_items)))
    )
    idx2name = class_index(model)
    results: dict[str, float] = {}

    for kind in PERTURBATIONS:
        y_true: list[str] = []
        y_pred: list[str] = []
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            imgs, trues = [], []
            for p, t in chunk:
                im = cv2.imread(p)
                if im is None:
                    continue
                imgs.append(perturb(im, kind))
                trues.append(t)
            if not imgs:
                continue
            preds = model.predict(imgs, imgsz=imgsz, verbose=False)
            for r, t in zip(preds, trues, strict=True):
                y_pred.append(idx2name[int(r.probs.top1)])
                y_true.append(t)

        results[kind] = float(accuracy_score(y_true, y_pred))
        print(f"[robustez:{tag}] {kind:10s} acc = {results[kind]:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_dir / "robustness.csv", index=False)
    robustness_latex(results, tag, out_dir / f"table_robustness_{tag}.tex")
    return results
