import cv2
import numpy as np
import pandas as pd
import pytest

from fruitclf.config import GroupingConfig
from fruitclf.data.grouping import assign_groups, name_prefix
from fruitclf.data.splitting import DegenerateSplitError, split_grouped


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("prod0_1.png", "prod0"),
        ("prod3_12.png", "prod3"),
        ("1_1_tomato_wb_7.jpg", "1_1_tomato_wb"),
        ("apple - 12.jpg", "apple"),
        ("img_jpg.rf.a1b2c3.jpg", "img"),
    ],
)
def test_name_prefix(filename, expected):
    assert name_prefix(filename) == expected


def test_name_prefix_keeps_distinct_items_distinct():
    """Remover contadores em passes repetidos colapsava prod0 e prod3 em 'prod'."""
    assert name_prefix("prod0_1.png") != name_prefix("prod3_1.png")


def _make_items(tmp_path, n_items=6, n_frames=5, seed=0):
    """Cria n_items objetos visualmente distintos, com n_frames quase idênticos."""
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(n_items):
        base = np.zeros((96, 96, 3), np.uint8)
        cv2.circle(base, (15 + item * 12, 30 + item * 9), 10 + item * 5,
                   (40 + item * 35, 220 - item * 30, 90 + item * 20), -1)
        cv2.rectangle(base, (2, 60), (94, 70 + item * 4),
                      (25 * item, 60, 210 - 25 * item), -1)
        for f in range(n_frames):
            noise = rng.integers(-5, 5, base.shape)
            img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            p = tmp_path / f"item{item}_{f}.png"
            cv2.imwrite(str(p), img)
            rows.append({"path": str(p), "filename": p.name, "cls": "a"})
    return pd.DataFrame(rows)


def test_assign_groups_adds_group_column(tmp_path):
    df = _make_items(tmp_path)
    out = assign_groups(df, "cls")
    assert "group" in out.columns
    assert len(out) == len(df)


def test_assign_groups_keeps_frames_of_same_item_together(tmp_path):
    """Frames do mesmo item nunca podem cair em grupos diferentes."""
    df = _make_items(tmp_path)
    out = assign_groups(df, "cls")
    out["item"] = out["filename"].str.split("_").str[0]
    for item, sub in out.groupby("item"):
        assert sub["group"].nunique() == 1, f"{item} ficou dividido em grupos"


def test_grouped_split_has_no_shared_groups(tmp_path):
    df = _make_items(tmp_path, n_items=12, n_frames=4)
    df["cls"] = ["a"] * 24 + ["b"] * 24
    grouped = assign_groups(df, "cls", GroupingConfig(ham_thresh=4))
    tr, va = split_grouped(grouped, "cls", n_splits=3)
    assert not (set(tr["group"]) & set(va["group"]))
    assert len(tr) + len(va) == len(grouped)


def test_split_grouped_rejects_degenerate_partition(tmp_path):
    """Um lado vazio passava em silêncio e só quebrava durante o treino."""
    df = _make_items(tmp_path, n_items=6, n_frames=3)
    df["group"] = 0  # todos no mesmo grupo: split impossível sem vazar
    with pytest.raises(DegenerateSplitError):
        split_grouped(df, "cls", n_splits=3)
