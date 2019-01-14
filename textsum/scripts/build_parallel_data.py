import os
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold

random.seed(2019)

if __name__ == "__main__":
    if not os.path.exists("textsum/data/"):
        os.mkdir("textsum/data/")
    if not os.path.exists("textsum/data/chinese_new"):
        os.mkdir("textsum/data/chinese_new/")

    df = pd.read_csv("data/tokenized/chinese_new/chinese_news.csv")
    y = df.tag
    skf = StratifiedKFold(n_splits=5)
    for tr_idx, te_idx in skf.split(df, y):
        tr_df = df.iloc[tr_idx]
        tmp_df = df.iloc[te_idx]
        break
    y = tmp_df.tag
    skf = StratifiedKFold(n_splits=2)
    for val_idx, te_idx in skf.split(tmp_df, y):
        val_df = tmp_df.iloc[val_idx]
        te_df = tmp_df.iloc[te_idx]
    print("Train size: ", tr_df.shape)
    print(tr_df.iloc[0]["headline"])
    print("Valid size: ", val_df.shape)
    print(val_df.iloc[0]["headline"])
    print("Test size: ", te_df.shape)
    print(te_df.iloc[0]["headline"])
    tr_df.to_csv("textsum/data/chinese_new/train.csv", index=False)
    val_df.to_csv("textsum/data/chinese_new/valid.csv", index=False)
    te_df.to_csv("textsum/data/chinese_new/test.csv", index=False)

    with open("textsum/data/chinese_new/train.src.txt", "w") as f:
        for content in tr_df.content.tolist():
            f.write(content.replace("\n", " </d> ") + "\n")
    with open("textsum/data/chinese_new/train.tgt.txt", "w") as f:
        for headline in tr_df.headline.tolist():
            f.write(headline.replace("\n", " </d> ") + "\n")

    with open("textsum/data/chinese_new/valid.src.txt", "w") as f:
        for content in val_df.content.tolist():
            f.write(content.replace("\n", " </d> ") + "\n")
    with open("textsum/data/chinese_new/valid.tgt.txt", "w") as f:
        for headline in val_df.headline.tolist():
            f.write(headline.replace("\n", " </d> ") + "\n")

    with open("textsum/data/chinese_new/test.src.txt", "w") as f:
        for content in te_df.content.tolist():
            f.write(content.replace("\n", " </d> ") + "\n")
    with open("textsum/data/chinese_new/test.tgt.txt", "w") as f:
        for headline in te_df.headline.tolist():
            f.write(headline.replace("\n", " </d> ") + "\n")