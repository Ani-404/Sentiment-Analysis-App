import os
import tempfile

# Fix Streamlit permissions issue on HF Spaces
os.environ["STREAMLIT_CONFIG_DIR"] = tempfile.mkdtemp()

import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Defaults to the project's own fine-tuned models. emotion-model is public;
# finbert-model is gated, so a public deployment needs either the repo ungated
# or an authorized HF_TOKEN available to the server. Override via env vars.
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "Ani-404/emotion-model")
FINANCIAL_MODEL = os.getenv("FINANCIAL_MODEL", "Ani-404/finbert-model")

# Configure page
st.set_page_config(
    page_title="SentText - Advanced Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize models
@st.cache_resource
def load_models():
    models = {}
    errors = {}
    for key, model_id in (('emotion', EMOTION_MODEL), ('financial', FINANCIAL_MODEL)):
        try:
            models[key] = pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
            )
        except Exception as exc:
            errors[key] = f"{model_id}: {exc}"
    return models, errors

# Prediction functions

def predict_emotions_real(text, model):
    results = model(text, top_k=None)
    scores_list = results[0] if results and isinstance(results[0], list) else results
    top = max(scores_list, key=lambda x: x['score'])
    return top['label'].lower(), top['score']


def analyze_financial_real(text, model):
    results = model(text)
    res = results[0]
    label = res['label'].lower()
    confidence = res['score']
    if 'positive' in label:
        score = confidence
        signal = 'BUY' if confidence>0.7 else 'HOLD'
    elif 'negative' in label:
        score = -confidence
        signal = 'SELL' if confidence>0.7 else 'HOLD'
    else:
        score = 0; signal='HOLD'
    return score, confidence, signal


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker):
    """Fetch 5-day price history; cached to reduce Yahoo rate-limiting."""
    return yf.Ticker(ticker).history(period='5d')

# Main UI

def main():
    st.title("SentText Analytics")
    models, errors = load_models()
    if errors:
        with st.sidebar:
            st.warning("Some models failed to load:")
            for key, msg in errors.items():
                st.caption(f"{key}: {msg}")
    tabs = st.tabs(["🎭 Emotion Analysis", "📈 Financial Analysis"])

    # Emotion
    with tabs[0]:
        text = st.text_area("Enter text:")
        if st.button("Analyze Emotion") and text:
            if 'emotion' in models:
                label, conf = predict_emotions_real(text, models['emotion'])
                st.write(f"**Emotion**: {label} | **Confidence**: {conf:.1%}")
            else:
                st.error("Emotion model not loaded.")

    # Financial
    with tabs[1]:
        col1, col2 = st.columns([1,2])
        with col1:
            ticker = st.text_input("Ticker:", value='AAPL')
            if st.button("Fetch Chart") and ticker.strip():
                try:
                    df = fetch_price_history(ticker.strip().upper())
                except Exception as exc:
                    df = None
                    if 'ratelimit' in type(exc).__name__.lower() or 'Too Many Requests' in str(exc):
                        st.warning("Yahoo Finance is rate-limiting the server right now. Please try again in a minute.")
                    else:
                        st.error(f"Couldn't fetch price data: {type(exc).__name__}")
                if df is not None and not df.empty:
                    fig = px.line(df, y='Close', title=f"{ticker.upper()} Closing Prices (5d)")
                    st.plotly_chart(fig)
                elif df is not None:
                    st.error("No data for that ticker.")
        with col2:
            fin_text = st.text_area("Enter financial text:")
            if st.button("Analyze Financial Sentiment") and fin_text:
                if 'financial' in models:
                    score, conf, signal = analyze_financial_real(fin_text, models['financial'])
                    st.write(f"**Sentiment Score**: {score:.2f} | **Confidence**: {conf:.1%}")
                    st.write(f"**Signal**: {signal}")
                else:
                    st.error("Financial model not loaded.")

if __name__ == '__main__':
    main()