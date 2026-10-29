import pandas as pd

from src.data_cleaning import remove_ambiguous_texts


def test_remove_ambiguous_texts():
    train_df = pd.DataFrame(
        {
            "medical_abstract": ["texto A", "texto B", "texto C"],
            "condition_label": [3, 1, 2],
        }
    )
    test_df = pd.DataFrame(
        {
            "medical_abstract": ["texto B", "texto C"],
            "condition_label": [4, 2],
        }
    )

    train_clean, test_clean = remove_ambiguous_texts(train_df, test_df)

    assert list(train_clean["medical_abstract"]) == ["texto A", "texto C"]
    assert list(test_clean["medical_abstract"]) == ["texto C"]
