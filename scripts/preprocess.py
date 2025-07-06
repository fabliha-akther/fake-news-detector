import pandas as pd
from pathlib import Path

def preprocess_kaggle(fake_path="data/Fake.csv", true_path="data/True.csv", output_path="data/final_dataset.csv"):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df.dropna(subset=["title", "text", "date"], inplace=True)
    df.rename(columns={"text": "content"}, inplace=True)
    df["id"] = range(1, len(df) + 1)
    df = df[["id", "title", "content", "date", "label"]]

    Path("data").mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed {len(df)} rows to {output_path}")

if __name__ == "__main__":
    preprocess_kaggle()
