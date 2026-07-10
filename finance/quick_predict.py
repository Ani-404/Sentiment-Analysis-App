"""
Light-weight predictor that fuses price momentum with model-based
financial sentiment.

Usage:
    from finance.quick_predict import predict_signal
    signal = predict_signal("AAPL", "Earnings were upbeat...")
    # -> "Buy", "Hold" or "Sell"
"""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from finance.processor import get_classifier, _label_to_score


def _yesterday_pct(ticker: str) -> float:
    """Return the most recent day's % change."""
    today = date.today()
    data = yf.download(
        ticker, start=today - timedelta(days=5), end=today, progress=False
    )
    if len(data) < 2:
        return 0.0
    # Recent yfinance returns MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[-2])
    return (close - prev) / prev * 100.0


def _sentiment_score(text: str) -> float:
    """Signed sentiment score in [-1, 1] weighted by model confidence."""
    classifier = get_classifier()
    result = classifier(text, truncation=True, max_length=128)[0]
    return _label_to_score(result["label"]) * float(result["score"])


def predict_signal(ticker: str, news_text: str) -> str:
    """Buy / Hold / Sell based on momentum + sentiment."""
    pct = _yesterday_pct(ticker)
    emo = _sentiment_score(news_text)
    blended = 0.6 * pct + 40 * emo  # scale sentiment to ~equal weight

    if blended > 2.0:
        return "Buy"
    if blended < -2.0:
        return "Sell"
    return "Hold"
