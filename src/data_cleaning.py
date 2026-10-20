import pandas as pd


def remove_ambiguous_texts(train_df, test_df):
    """Remove textos que aparecem com mais de um rótulo (treino + teste somados).

    Devolve (train_clean, test_clean).
    """
    cols = ["medical_abstract", "condition_label"]

    train = train_df[cols].assign(split="train")
    test = test_df[cols].assign(split="test")

    all_df = pd.concat([train, test], ignore_index=True)

    n_rotulos = all_df.groupby("medical_abstract")["condition_label"].transform(
        "nunique"
    )
    limpo = all_df[n_rotulos == 1]

    train_clean = limpo[limpo["split"] == "train"].drop(columns="split")
    test_clean = limpo[limpo["split"] == "test"].drop(columns="split")

    return train_clean, test_clean
