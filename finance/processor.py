# processor.py
# Processes earnings call transcripts into sentiment features using the
# hosted HuggingFace model (Ani-404/finbert-model), and fetches post-earnings
# stock returns as prediction targets.

import os
import re
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf

# Financial sentiment model. Defaults to the project's fine-tuned model
# (gated on HF, so needs an authorized token or the repo ungated). Override
# with FINANCIAL_MODEL to point at a different/public model.
MODEL_NAME = os.getenv("FINANCIAL_MODEL", "Ani-404/finbert-model")


@lru_cache(maxsize=1)
def get_classifier():
    """Load and cache the financial sentiment classifier pipeline."""
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def ingest_transcripts(project_root: str) -> pd.DataFrame:
    """Load the sample earnings-call transcripts."""
    file_path = os.path.join(project_root, "finance", "sample_transcripts_advanced.csv")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find transcript data at '{file_path}'.")


def preprocess_and_split(text):
    """Clean text and split it into sentences."""
    if not isinstance(text, str):
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _label_to_score(label: str) -> int:
    """Map a sentiment label to a simplified -1 / 0 / +1 score."""
    l = label.lower()
    if "pos" in l:
        return 1
    if "neg" in l:
        return -1
    return 0


def score_sentences(sentences, classifier=None):
    """Run the classifier on sentences and return a -1/0/+1 score each."""
    if not sentences:
        return []
    classifier = classifier or get_classifier()
    results = classifier(sentences, truncation=True, max_length=128)
    return [_label_to_score(r["label"]) for r in results]


def aggregate_sentiment_features(scores):
    """Aggregate sentence scores into summary features."""
    if not scores:
        return {"mean_score": 0.0, "positive_ratio": 0.0, "negative_ratio": 0.0}

    total_sentences = len(scores)
    positive_sentences = sum(1 for s in scores if s > 0)
    negative_sentences = sum(1 for s in scores if s < 0)

    return {
        "mean_score": float(np.mean(scores)),
        "positive_ratio": positive_sentences / total_sentences,
        "negative_ratio": negative_sentences / total_sentences,
    }


def get_stock_returns(ticker, earnings_date_str):
    """Fetch 1-day and 5-day returns following an earnings date."""
    try:
        earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        start_date = earnings_date - timedelta(days=1)
        end_date = earnings_date + timedelta(days=10)

        stock_data = yf.download(
            ticker, start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        if stock_data.empty:
            return None

        # Recent yfinance returns MultiIndex columns even for a single ticker.
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        price_on_date_series = stock_data.loc[stock_data.index >= earnings_date]
        if price_on_date_series.empty:
            return None
        price_on_date = float(price_on_date_series.iloc[0]["Close"])

        post_earnings_prices = stock_data.loc[stock_data.index > earnings_date]
        if len(post_earnings_prices) < 5:
            return None

        price_1d_after = float(post_earnings_prices.iloc[0]["Close"])
        price_5d_after = float(post_earnings_prices.iloc[4]["Close"])

        return {
            "return_1d": (price_1d_after - price_on_date) / price_on_date,
            "return_5d": (price_5d_after - price_on_date) / price_on_date,
        }
    except Exception:
        return None
