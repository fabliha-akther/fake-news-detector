import pandas as pd
from pathlib import Path
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(text)["compound"]

def extract_features(input_path="data/final_dataset.csv", output_path="data/features.csv"):
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        if "target" in df.columns:
            df["label"] = df["target"]
        else:
            raise ValueError("No label column found")

    df["content"] = df["content"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Temporal Features
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df.loc[df["date"].isna(), ["hour", "dayofweek", "is_weekend"]] = -1

    # Content Features
    df["content_length"] = df["content"].apply(len)
    df["num_exclamations"] = df["content"].str.count("!")
    df["num_questions"] = df["content"].str.count(r"\?")
    df["num_uppercase_words"] = df["content"].apply(lambda x: sum(w.isupper() for w in x.split()))
    df["num_links"] = df["content"].apply(lambda x: len(re.findall(r"http\S+", x)))
    df["num_hashtags"] = df["content"].str.count("#")

    # Keyword hits related to fake news
    df["keyword_hits"] = df["content"].apply(
        lambda x: sum(kw in x.lower() for kw in ["breaking", "urgent", "shocking", "fake"])
    )

    tqdm.pandas(desc="Calculating sentiment scores")

    # sentiment analysis
    df["sentiment_score"] = df["content"].progress_apply(lambda x: get_sentiment(x))

    Path("data").mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    extract_features()
