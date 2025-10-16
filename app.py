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
    try:
        # Remove use_auth_token, rely on environment login
        models['emotion'] = pipeline(
            "text-classification",
            model="Ani-404/emotion-model",
            tokenizer="Ani-404/emotion-model"
        )
        models['financial'] = pipeline(
            "text-classification", 
            model="Ani-404/finbert-model",
            tokenizer="Ani-404/finbert-model"
        )
    except Exception:
        models = {}
    return models

# Prediction functions

def predict_emotions_real(text, model):
    results = model(text, return_all_scores=True)
    scores_list = results[0]
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

# Main UI

def main():
    st.title("SentText Analytics")
    models = load_models()
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
            if st.button("Fetch Chart"):
                df = yf.Ticker(ticker).history(period='5d')
                if not df.empty:
                    fig = px.line(df, y='Close', title=f"{ticker} Closing Prices (5d)")
                    st.plotly_chart(fig)
                else:
                    st.error("No data for ticker.")
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