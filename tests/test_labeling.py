import pandas as pd
import pytest

from fruitclf.data.labeling import build_labels, extract_bag, extract_label


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("1_1_tomato_wb_12.jpg", "tomato"),
        ("apple - 12.jpg", "apple"),
        ("raspberry-3.png", "raspberry"),
        ("lemon_wob.jpg", "lemon"),
    ],
)
def test_extract_label(filename, expected):
    assert extract_label(filename) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("1_1_tomato_wb_12.jpg", "Yes"),
        ("1_1_tomato_wob_12.jpg", "No"),
        ("apple-wb-3.png", "Yes"),
        ("grapes wb 1.jpg", "Yes"),
        ("chilli_wob.jpg", "No"),
        ("raspberry - 12.jpg", "Unknown"),
    ],
)
def test_extract_bag(filename, expected):
    assert extract_bag(filename) == expected


def test_extract_bag_handles_split_tokens():
    """Variações como 'wo_b' caíam no default na versão anterior."""
    assert extract_bag("banana_wo_b_5.jpg") == "No"
    assert extract_bag("banana_w_b_5.jpg") == "Yes"


def test_extract_bag_prefers_without_over_with():
    """'wob' nunca deve ser lido como 'wb'."""
    assert extract_bag("tomato_wob_1.jpg") == "No"


def test_build_labels_taxonomies():
    df = pd.DataFrame(
        {
            "filename": [
                "1_1_tomato_wb_1.jpg",
                "1_1_tomato_wob_1.jpg",
                "apple - 2.jpg",
            ],
            "path": ["a", "b", "c"],
        }
    )
    out = build_labels(df)
    assert list(out["label_8"]) == ["tomato", "tomato", "apple"]
    assert list(out["label_14"]) == [
        "tomato_with_bag",
        "tomato_without_bag",
        "apple",
    ]
