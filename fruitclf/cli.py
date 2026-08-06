"""Interface de linha de comando.

    python -m fruitclf inspect-groups --config 14
    python -m fruitclf audit --config 14
    python -m fruitclf run --config both --split-mode both --epochs 50
"""

from __future__ import annotations

import argparse
import json

from fruitclf.config import (
    OUT_ROOT,
    TAXONOMIES,
    ExperimentConfig,
    GroupingConfig,
    TrainConfig,
    set_seed,
)
from fruitclf.data.audit import agreement_rate, export_bag_audit
from fruitclf.data.grouping import assign_groups, inspect_groups
from fruitclf.data.ingest import load_raw
from fruitclf.data.labeling import build_labels, describe
from fruitclf.evaluation.reporting import main_table_latex
from fruitclf.experiment import run_experiment


def _load_dataset(root: str | None = None):
    df = build_labels(load_raw(root))
    print(describe(df))
    return df


def _taxonomies(choice: str) -> list[tuple[str, str]]:
    items = list(TAXONOMIES.items())
    return items if choice == "both" else [(k, v) for k, v in items if k == choice]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", choices=["8", "14", "both"], default="both")
    p.add_argument("--dataset-root", default=None, help="pular o download do Kaggle")


def _add_grouping(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ham-thresh", type=int, default=8)
    p.add_argument("--no-prefix-merge", action="store_true")


def _grouping_cfg(args) -> GroupingConfig:
    return GroupingConfig(
        ham_thresh=args.ham_thresh, use_prefix=not args.no_prefix_merge
    )


def cmd_inspect_groups(args) -> None:
    df = _load_dataset(args.dataset_root)
    for tag, col in _taxonomies(args.config):
        grouped = assign_groups(df, col, _grouping_cfg(args))
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(OUT_ROOT / f"groups_{tag}.csv", index=False)
        inspect_groups(grouped, OUT_ROOT / f"groups_preview_{tag}.png")


def cmd_audit(args) -> None:
    df = _load_dataset(args.dataset_root)
    sheet = OUT_ROOT / "audit_bag_labels" / "audit_sheet.csv"
    if args.score:
        result = agreement_rate(sheet)
        (OUT_ROOT / "audit_bag_labels" / "agreement.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    else:
        export_bag_audit(df, OUT_ROOT / "audit_bag_labels", n=args.n)


def cmd_run(args) -> None:
    df = _load_dataset(args.dataset_root)
    modes = ["random", "grouped"] if args.split_mode == "both" else [args.split_mode]
    train_cfg = TrainConfig(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    payloads = []
    for tag, col in _taxonomies(args.config):
        df_tax = df
        if "grouped" in modes:
            df_tax = assign_groups(df, col, _grouping_cfg(args))
            df_tax.to_csv(OUT_ROOT / f"groups_{tag}.csv", index=False)
        for mode in modes:
            cfg = ExperimentConfig(
                tag=tag,
                label_col=col,
                split_mode=mode,
                train=train_cfg,
                robustness_subset=args.robustness_subset,
            )
            payloads.append(run_experiment(df_tax, cfg))

    main_table_latex(payloads, OUT_ROOT / "table_main.tex")
    (OUT_ROOT / "all_results.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8"
    )
    print(f"\n[fim] tabelas LaTeX e metricas em {OUT_ROOT}/")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fruitclf")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect-groups", help="exportar grid dos grupos formados")
    _add_common(p)
    _add_grouping(p)
    p.set_defaults(func=cmd_inspect_groups)

    p = sub.add_parser("audit", help="exportar ou pontuar a auditoria de embalagem")
    _add_common(p)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--score", action="store_true", help="calcular a concordancia")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("run", help="treinar e avaliar")
    _add_common(p)
    _add_grouping(p)
    p.add_argument(
        "--split-mode", choices=["random", "grouped", "both"], default="both"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--robustness-subset", type=int, default=None)
    p.set_defaults(func=cmd_run)

    return ap


def main() -> None:
    set_seed()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
