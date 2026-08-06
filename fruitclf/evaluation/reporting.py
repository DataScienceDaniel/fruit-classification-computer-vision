"""Geração de figuras e tabelas LaTeX prontas para o artigo.

Cada configuração escreve no seu próprio diretório, o que impede que as
tabelas de 8 e 14 classes sejam confundidas entre si.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _escape(label: str) -> str:
    return label.replace("_", r"\_")


def plot_confusion(
    cm: np.ndarray, labels: list[str], out_png: Path, normalize: bool = False
) -> None:
    """Matriz de confusão. Prefira a versão normalizada com classes desbalanceadas."""
    M = cm.astype(float)
    if normalize:
        M = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 0.55), max(5, len(labels) * 0.5))
    )
    im = ax.imshow(
        M,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize else M.max(),
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    thresh = M.max() / 2.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                txt = f"{M[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if M[i, j] > thresh else "black",
                )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _table(caption: str, label: str, colspec: str, header: str, body: list[str]) -> str:
    return "\n".join(
        [
            r"\begin{table}[h!]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{" + colspec + "}",
            r"\toprule",
            header,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def per_class_latex(labels, pr, rc, f1, support, caption, label, out_tex: Path) -> None:
    body = [
        f"\\textit{{{_escape(lab)}}} & {p:.3f} & {r:.3f} & {f:.3f} & {int(s)} \\\\"
        for lab, p, r, f, s in zip(labels, pr, rc, f1, support, strict=True)
    ]
    body += [
        r"\midrule",
        f"\\textbf{{Macro avg}} & {np.mean(pr):.3f} & {np.mean(rc):.3f} & "
        f"{np.mean(f1):.3f} & {int(np.sum(support))} \\\\",
    ]
    header = (
        r"\textbf{Class} & \textbf{Prec.} & \textbf{Rec.} & "
        r"\textbf{F1} & \textbf{Support} \\"
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        _table(caption, label, "l c c c c", header, body), encoding="utf-8"
    )


def latency_latex(stats: list[dict], out_tex: Path) -> None:
    body = [
        f"{s['device'].upper()} & {s['mean_ms']:.1f} $\\pm$ {s['std_ms']:.1f} & "
        f"{s['p50_ms']:.1f} & {s['p95_ms']:.1f} & {s['max_ms']:.1f} \\\\"
        for s in stats
    ]
    caption = (
        "Single-image inference latency (batch size 1, $224\\times224$ input, "
        "50 timed runs after 10 warm-up runs)."
    )
    header = (
        r"\textbf{Device} & \textbf{Mean (ms)} & \textbf{P50} & "
        r"\textbf{P95} & \textbf{Max} \\"
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        _table(caption, "tab:latency", "l c c c c", header, body), encoding="utf-8"
    )


def robustness_latex(results: dict, tag: str, out_tex: Path) -> None:
    body = [f"{k.capitalize()} & {v:.3f} \\\\" for k, v in results.items()]
    caption = f"Validation accuracy under controlled perturbations ({tag})."
    header = r"\textbf{Perturbation} & \textbf{Accuracy} \\"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        _table(caption, f"tab:robustness_{tag}", "l c", header, body), encoding="utf-8"
    )


def main_table_latex(payloads: list[dict], out_tex: Path) -> None:
    """Tabela principal: split aleatório contra split por grupo.

    A queda de acurácia entre as duas linhas é a medida do vazamento, e é a
    resposta direta à crítica central dos revisores.
    """
    body = []
    for p in payloads:
        m, s = p["metrics"], p["split"]
        mode = "random (img)" if s["split_mode"] == "random" else "grouped"
        body.append(
            f"{s['n_classes']} classes & {mode} & {m['accuracy']:.3f} & "
            f"{m['macro_precision']:.3f} & {m['macro_recall']:.3f} & "
            f"{m['macro_f1']:.3f} \\\\"
        )
    caption = (
        "Validation performance under an image-level random split (leaky "
        "baseline) and a group-aware split in which all near-duplicate frames "
        "of the same physical item are kept on the same side."
    )
    header = (
        r"\textbf{Setting} & \textbf{Split} & \textbf{Acc.} & "
        r"\textbf{Macro P} & \textbf{Macro R} & \textbf{Macro F1} \\"
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(
        _table(caption, "tab:main", "l c c c c c", header, body), encoding="utf-8"
    )
