# Automatic Fruit and Vegetable Classification — SBrT 2026

Pipeline de classificação de frutas e vegetais para sistemas de pesagem inteligentes, usando YOLOv8s sobre o dataset *Fruits & Vegetable Detection for YOLOv4*.

Esta é a versão revisada após o parecer do SBrT 2026 (paper #1571281644). A mudança principal é metodológica: o dataset de origem foi montado para **detecção** e contém sequências de frames quase idênticos do mesmo item físico, então uma partição no nível da imagem coloca frames praticamente iguais nos dois lados e o classificador atinge acurácia perfeita memorizando itens. O pipeline agora agrupa quase-duplicatas antes de particionar, e roda as duas estratégias lado a lado para que a diferença seja reportável.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Uso

```bash
# 1. Conferir os grupos visualmente antes de treinar (rápido, sem treino)
python -m fruitclf inspect-groups --config 14

# 2. Exportar amostra para auditoria manual dos rótulos de embalagem
python -m fruitclf audit --config 14 --n 120
#    ... preencher a coluna bag_verdadeiro em outputs/audit_bag_labels/audit_sheet.csv
python -m fruitclf audit --score

# 3. Treinar e avaliar as quatro configurações (8/14 classes × random/grouped)
python -m fruitclf run --config both --split-mode both --epochs 50
```

## Estrutura

```
fruitclf/
├── config.py              # constantes e dataclasses de configuração
├── cli.py                 # interface de linha de comando
├── experiment.py          # orquestra split → treino → avaliação
├── training.py            # fine-tuning do YOLOv8s
├── data/
│   ├── ingest.py          # download e varredura do dataset
│   ├── labeling.py        # extração de rótulos e do atributo de embalagem
│   ├── grouping.py        # detecção de quase-duplicatas (dHash + componentes conexas)
│   ├── splitting.py       # partições random e group-aware
│   └── audit.py           # auditoria manual dos rótulos wb/wob
└── evaluation/
    ├── metrics.py         # inferência e métricas por classe
    ├── latency.py         # benchmark em CPU e GPU
    ├── robustness.py      # perturbações controladas
    └── reporting.py       # figuras e tabelas LaTeX
```

## Saídas

Cada configuração escreve em `outputs/{8,14}_{random,grouped}/`:

| Arquivo | Conteúdo |
|---|---|
| `results.json` | métricas, split, latência, robustez, tamanho do modelo |
| `per_class_metrics.csv` | precision / recall / F1 por classe |
| `confusion_matrix{,_norm}.png` | matriz de confusão (contagens e normalizada) |
| `val_predictions.csv` | predição e confiança por imagem |
| `robustness.csv` | acurácia sob cada perturbação |
| `table_*.tex` | tabelas prontas para `\input` no Overleaf |

No nível raiz, `outputs/table_main.tex` compara as duas estratégias de split — é a tabela que responde à crítica central dos revisores.

## Ajuste do agrupamento

Rode `inspect-groups` antes de treinar. O grid gerado mostra os maiores grupos; cada linha deve conter frames do **mesmo item físico**.

- Linhas misturando itens distintos → `--ham-thresh` menor (mais rigoroso).
- Itens obviamente iguais em linhas diferentes → `--ham-thresh` maior.
- Prefixos de nome de arquivo atrapalhando → `--no-prefix-merge`.

## Testes

```bash
pytest tests/ -q
ruff check fruitclf tests
```
