import pandas as pd
import re
import yfinance as yf
from datetime import timedelta
import numpy as np
import torch

def ingest_transcripts(file_path='finance/sample_transcripts_advanced.csv'):
    """Ingests earnings call transcripts from the advanced CSV file."""
    return pd.read_csv(file_path)

def preprocess_and_split(transcript_text):
    """Splits raw text into clean, individual sentences."""
    if not isinstance(transcript_text, str):
        return []
    sentences = re.split(r'(?<=[.!?]) +', transcript_text)
    cleaned_sentences = [s.strip() for s in sentences if s.strip()]
    return cleaned_sentences

def get_sentiment_vectors(sentences, model, tokenizer):
    """Runs FinBERT on sentences to get full emotion probability vectors."""
    if not sentences:
        return []
    
    vectors = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256)
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            vectors.append(probabilities)
    return vectors

def aggregate_vectors_to_features(vectors, model, prefix=''):
    """Aggregates sentence vectors into call-level features."""
    if not vectors:
        # Returning a dictionary with zero values for all emotions if no vectors
        return {f"{prefix}{model.config.id2label[i]}_mean": 0 for i in range(model.config.num_labels)}

    vectors_np = np.array(vectors)
    # Calculating the mean probability for each emotion across all sentences
    mean_sentiments = np.mean(vectors_np, axis=0)
    
    # Creating a feature dictionary
    features = {f"{prefix}{model.config.id2label[i]}_mean": mean_sentiments[i] for i in range(len(mean_sentiments))}
    return features

def get_stock_returns(ticker, earnings_date):
    """Fetches stock prices and calculates 1-day and 5-day returns."""
    earnings_date = pd.to_datetime(earnings_date)
    start_date = earnings_date - timedelta(days=1)
    end_date = earnings_date + timedelta(days=7)
    
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if stock_data.empty: return None

    try:
        price_on_earnings_date = stock_data.loc[stock_data.index.date == earnings_date.date()]['Adj Close'].iloc[0]
    except IndexError:
        try:
            price_on_earnings_date = stock_data[stock_data.index > earnings_date]['Adj Close'].iloc[0]
        except IndexError: return None

    try:
        price_1_day_after = stock_data[stock_data.index > earnings_date]['Adj Close'].iloc[0]
        price_5_days_after = stock_data[stock_data.index > earnings_date]['Adj Close'].iloc[4]
        
        return_1d = (price_1_day_after - price_on_earnings_date) / price_on_earnings_date
        return_5d = (price_5_days_after - price_on_earnings_date) / price_on_earnings_date
        return {'return_1d': return_1d, 'return_5d': return_5d}
    except IndexError: return None