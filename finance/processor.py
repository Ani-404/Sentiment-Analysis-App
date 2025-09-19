# processor.py
# This module processes earnings call transcripts to extract sentiment features using FinBERT 

import pandas as pd
import re
import yfinance as yf
from datetime import timedelta
import numpy as np
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def get_sentiment_vectors(texts: list[str]) -> list[list[float]]:
    """Return TF-IDF vectors for a list of texts."""
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(texts).toarray()

def ingest_transcripts(project_root: str):
    """
    Ingests the NEW, simplified sample earnings call transcripts.
    """
    file_path = os.path.join(project_root, "finance", "sample_transcripts.csv")
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find transcript data at '{file_path}'.")

def preprocess_and_split(text):
    """Cleans and splits text into sentences."""
    if not isinstance(text, str):
        return []
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

@torch.no_grad()
def get_sentiment_scores(sentences, model, tokenizer):
    """
    Runs the model on sentences and returns a simplified -1, 0, +1 score per sentence.
    """
    if not sentences:
        return []
        
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    score_map = {
        'joy': 1, 'surprise': 1,
        'anger': -1, 'fear': -1, 'sadness': -1,
        'neutral': 0, 'shame': 0, 'disgust': 0
    }
    
    labels = [model.config.id2label[p.item()] for p in predictions]
    scores = [score_map.get(label, 0) for label in labels]
    return scores

def aggregate_sentiment_features(scores):
    """Aggregates a list of scores into features like mean score and positive ratio."""
    if not scores:
        return {'mean_score': 0, 'positive_ratio': 0, 'negative_ratio': 0}
    
    total_sentences = len(scores)
    positive_sentences = sum(1 for s in scores if s > 0)
    negative_sentences = sum(1 for s in scores if s < 0)
    
    return {
        'mean_score': np.mean(scores),
        'positive_ratio': positive_sentences / total_sentences,
        'negative_ratio': negative_sentences / total_sentences
    }

def get_stock_returns(ticker, earnings_date_str):
    """
    More robustly fetches stock prices and calculates 1-day and 5-day returns.
    """
    try:
        earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d')
        start_date = earnings_date - timedelta(days=1)
        end_date = earnings_date + timedelta(days=10)
        
        stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if stock_data.empty:
            return None

        price_on_date_series = stock_data.loc[stock_data.index >= earnings_date]
        if price_on_date_series.empty: return None
        # --- KEY CHANGE: Use 'Close' instead of 'Adj Close' ---
        price_on_date = price_on_date_series.iloc[0]['Close']
        
        post_earnings_prices = stock_data.loc[stock_data.index > earnings_date]
        if len(post_earnings_prices) < 5: return None

        # --- KEY CHANGE: Use 'Close' instead of 'Adj Close' ---
        price_1d_after = post_earnings_prices.iloc[0]['Close']
        price_5d_after = post_earnings_prices.iloc[4]['Close']
        # --- END OF CHANGE ---
        
        return {
            'return_1d': (price_1d_after - price_on_date) / price_on_date,
            'return_5d': (price_5d_after - price_on_date) / price_on_date
        }
    except Exception:
        return None

