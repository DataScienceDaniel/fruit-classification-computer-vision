"""Inferência sobre o conjunto de validação e métricas de classificação."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fruitclf.config import IMG_EXTS
from fruitclf.evaluation.reporting import per_class_latex, plot_confusion


def class_index(model) -> dict[int, str]:
    names = model.names
    return names if isinstance(names, dict) else dict(enumerate(names))


def list_val_images(val_root: Path) -> list[tuple[str, str]]:
    """Lista pares (caminho, rótulo verdadeiro) do diretório de validação."""
    items = []
    for cdir in sorted(val_root.iterdir()):
        if cdir.is_dir():
            for img in sorted(cdir.rglob("*.*")):
                if img.suffix.lower() in IMG_EXTS:
                    items.append((str(img), cdir.name))
    return items


def predict_all(
    model,
    items: list[tuple[str, str]],
    imgsz: int = 224,
    device: str | None = None,
    batch: int = 32,
) -> tuple[list[str], list[str], list[dict]]:
    """Inferência em lote sobre pares (caminho, rótulo verdadeiro)."""
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []
    idx2name = class_index(model)

    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        paths = [p for p, _ in chunk]
        results = model.predict(paths, imgsz=imgsz, verbose=False, device=device)
        for (p, true_label), r in zip(chunk, results, strict=True):
            probs = r.probs.data.cpu().numpy()
            pred_label = idx2name[int(np.argmax(probs))]
            y_true.append(true_label)
            y_pred.append(pred_label)
            rows.append(
                {
                    "path": p,
                    "true": true_label,
                    "pred": pred_label,
                    "conf": float(np.max(probs)),
                }
            )
    return y_true, y_pred, rows


def evaluate(model, val_items, out_dir: Path, tag: str, imgsz: int = 224) -> dict:
    """Avalia e grava métricas por classe, matriz de confusão e predições."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, rows = predict_all(model, val_items, imgsz=imgsz)
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pd.DataFrame(rows).to_csv(out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(
        {"class": labels, "precision": pr, "recall": rc, "f1": f1, "support": sup}
    ).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out_dir / "confusion_matrix.csv"
    )

    plot_confusion(cm, labels, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion(cm, labels, out_dir / "confusion_matrix_norm.png", normalize=True)
    per_class_latex(
        labels,
        pr,
        rc,
        f1,
        sup,
        caption=f"Per-class performance ({tag}).",
        label=f"tab:perclass_{tag}",
        out_tex=out_dir / f"table_perclass_{tag}.tex",
    )

    print(f"\n=== {tag} ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    errors = [(r["true"], r["pred"]) for r in rows if r["true"] != r["pred"]]
    top_conf = Counter(errors).most_common(10)

    return {
        "accuracy": float(acc),
        "macro_precision": float(np.mean(pr)),
        "macro_recall": float(np.mean(rc)),
        "macro_f1": float(np.mean(f1)),
        "n_val_images": len(y_true),
        "n_classes": len(labels),
        "top_confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in top_conf
        ],
    }
