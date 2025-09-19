"""
Light-weight predictor that fuses price momentum with
FinBERT-emotional sentiment.

Usage:
    from finance.quick_predict import predict_signal
    signal = predict_signal("AAPL", "Earnings were upbeat…")
    # -> "Buy", "Hold" or "Sell"
"""
from datetime import date, timedelta
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent.parent / "finance" / "finbert_large_emotion_model"

_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
EMOTION_WEIGHTS = {
    "Joy":  +1.0,
    "Trust":+0.8,
    "Surprise":+0.2,
    "Neutral":0.0,
    "Fear":  -0.8,
    "Sadness":-1.0,
    "Anger": -0.6,
    "Disgust":-0.9,
}

def _load_finbert():
    global _tokenizer, _model
    if _tokenizer is None:
        MODEL = "Ani-404/FinBERT_large_emotional"     # adjust if repo name differs
        _tokenizer = AutoTokenizer.from_pretrained(MODEL)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return _tokenizer, _model

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _yesterday_pct(ticker: str) -> float:
    """Return yesterday's % change."""
    today = date.today()
    yday  = today - timedelta(days=1)
    data  = yf.download(ticker, start=yday - timedelta(days=2), end=today, progress=False)
    if data.empty < 2:
        return 0.0
    close, prev = data["Close"].iloc[-1], data["Close"].iloc[-2]
    return (close - prev) / prev * 100.0

def _emotion_score(text: str) -> float:
    tok, mod = _load_finbert()
    logits = mod(**tok(text, return_tensors="pt", truncation=True)).logits
    probs  = torch.softmax(logits, dim=1).detach().cpu().numpy().flatten()
    labels = list(EMOTION_WEIGHTS.keys())
    return float(np.dot(probs, [EMOTION_WEIGHTS[l] for l in labels]))

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def predict_signal(ticker: str, news_text: str) -> str:
    """Buy / Hold / Sell based on momentum + emotion."""
    pct      = _yesterday_pct(ticker)
    emo      = _emotion_score(news_text)
    blended  = 0.6 * pct + 40 * emo          # scale sentiment to ~equal weight

    if blended > 2.0:
        return "Buy"
    if blended < -2.0:
        return "Sell"
    return "Hold"
