from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Settings
from .text import normalize_tweet

AIRLINE_NAME = "Twitter US Airline Sentiment"
TWEETEVAL_NAME = "TweetEval Sentiment"
S140_NAME = "Sentiment140"


def _add_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["text"] = result["text"].fillna("").astype(str)
    result["clean_text"] = result["text"].map(normalize_tweet)
    result["dataset_row_id"] = np.arange(len(result))
    return result


def load_airline(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding_errors="replace")
    required = {"text", "airline_sentiment", "airline"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Airline file is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"airline_sentiment": "label"})
    return _add_common_columns(frame)


def load_tweeteval(data_dir: Path) -> Dict[str, pd.DataFrame]:
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    result = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(data_dir / f"tweet_eval_{split}.csv", encoding_errors="replace")
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"TweetEval {split} must contain text and label")
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").map(label_map)
        result[split] = _add_common_columns(frame)
    return result


def load_sentiment140(path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    names = ["sentiment", "tweet_id", "date", "query", "username", "text"]
    frame = pd.read_csv(path, header=None, names=names, encoding="latin-1", engine="python")
    frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="coerce")
    frame = frame[frame["sentiment"].isin([0, 4])].copy()
    frame["label"] = frame["sentiment"].map({0: "negative", 4: "positive"})
    if sample_size and sample_size < len(frame):
        per_class = sample_size // 2
        frame = pd.concat([
            frame[frame["label"] == label].sample(per_class, random_state=seed)
            for label in ("negative", "positive")
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return _add_common_columns(frame)


def load_datasets(settings: Settings) -> Dict[str, object]:
    return {
        AIRLINE_NAME: load_airline(settings.data_dir / "Tweets.csv"),
        TWEETEVAL_NAME: load_tweeteval(settings.data_dir),
        S140_NAME: load_sentiment140(
            settings.data_dir / "training.1600000.processed.noemoticon.csv",
            settings.s140_sample_size,
            settings.seed,
        ),
    }


def build_splits(datasets: Dict[str, object], settings: Settings) -> Dict[str, Dict[str, pd.DataFrame]]:
    splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name, value in datasets.items():
        if isinstance(value, dict):
            splits[name] = {key: frame.reset_index(drop=True) for key, frame in value.items()}
            continue
        train, held_out = train_test_split(
            value, test_size=settings.test_size + settings.validation_size,
            random_state=settings.seed, stratify=value["label"],
        )
        relative_test = settings.test_size / (settings.test_size + settings.validation_size)
        validation, test = train_test_split(
            held_out, test_size=relative_test, random_state=settings.seed, stratify=held_out["label"],
        )
        splits[name] = {"train": train, "validation": validation, "test": test}
    return splits


def quality_summary(splits: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for name, split_map in splits.items():
        frame = pd.concat(split_map.values(), ignore_index=True)
        rows.append({
            "Dataset": name,
            "Rows": len(frame),
            "Classes": int(frame["label"].nunique()),
            "Missing text": int(frame["text"].isna().sum()),
            "Duplicate raw text": int(frame["text"].duplicated().sum()),
            "Duplicate normalized text": int(frame["clean_text"].duplicated().sum()),
            "Mean words": float(frame["clean_text"].str.split().str.len().mean()),
        })
    return pd.DataFrame(rows)
