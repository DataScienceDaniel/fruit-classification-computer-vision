"""Orquestração de um experimento completo: split, treino e avaliação."""

from __future__ import annotations

import json

import pandas as pd

from fruitclf.config import ExperimentConfig
from fruitclf.data.splitting import (
    materialize,
    split_grouped,
    split_random,
    split_summary,
)
from fruitclf.evaluation.latency import benchmark_all_devices
from fruitclf.evaluation.metrics import evaluate, list_val_images
from fruitclf.evaluation.reporting import latency_latex
from fruitclf.evaluation.robustness import robustness_sweep
from fruitclf.training import load_checkpoint, model_size_mb, train_classifier


def run_experiment(df: pd.DataFrame, cfg: ExperimentConfig) -> dict:
    """Executa um experimento e grava ``results.json`` no diretório da config."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.split_mode == "grouped":
        tr, va = split_grouped(df, cfg.label_col)
    else:
        tr, va = split_random(df, cfg.label_col)

    split_info = split_summary(df, tr, va, cfg.label_col, cfg.split_mode)
    split_info.update(
        {
            "epochs": cfg.train.epochs,
            "imgsz": cfg.train.imgsz,
            "batch": cfg.train.batch,
            "seed": cfg.train.seed,
        }
    )

    dataset_dir = materialize(tr, va, cfg.label_col, cfg.dataset_dir)
    ckpt, _ = train_classifier(dataset_dir, cfg.out_dir, cfg.train)
    model = load_checkpoint(ckpt)

    val_items = list_val_images(dataset_dir / "val")
    metrics = evaluate(model, val_items, cfg.out_dir, cfg.name, imgsz=cfg.train.imgsz)
    robustness = robustness_sweep(
        model,
        val_items,
        cfg.out_dir,
        cfg.name,
        imgsz=cfg.train.imgsz,
        max_images=cfg.robustness_subset,
    )

    latency = benchmark_all_devices(ckpt, val_items[0][0], imgsz=cfg.train.imgsz)
    latency_latex(latency, cfg.out_dir / "table_latency.tex")

    payload = {
        "tag": cfg.tag,
        "split": split_info,
        "metrics": metrics,
        "robustness": robustness,
        "latency": latency,
        "model_size_mb": model_size_mb(ckpt),
        "checkpoint": ckpt,
    }
    (cfg.out_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[ok] resultados em {cfg.out_dir / 'results.json'}")
    return payload
