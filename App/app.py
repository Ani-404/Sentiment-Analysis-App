# app.py

# Core Pkgs
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_option_menu import option_menu
import os
import plotly.express as px

# This brings in the logic from your new 'finance' directory
from finance.processor import (
    ingest_transcripts,
    preprocess_and_split,
    get_sentiment_vectors,
    aggregate_vectors_to_features,
    get_stock_returns
)
from finance.analysis import run_prediction_model

# --- Configuration and Model Loading ---

# Set page configuration
st.set_page_config(
    page_title="SentText - Advanced Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """
    Loads both the general emotion model and the fine-tuned financial model.
    This function will be run only once.
    """
    models = {}
    try:
        # --- Determine Project Root Correctly ---
        # This handles running the script from 'App/' or the project's root directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(app_dir) == 'App':
            project_root = os.path.dirname(app_dir)
        else:
            project_root = app_dir

        # --- Load General Emotion Model (for the first tab) ---
        general_model_path = os.path.join(project_root, "Models", "sentiment_model_distilbert")
        st.info(f"Loading general model from: {general_model_path}")
        models['general_tokenizer'] = AutoTokenizer.from_pretrained(general_model_path)
        models['general_model'] = AutoModelForSequenceClassification.from_pretrained(general_model_path)
        
        # --- Load Fine-Tuned Financial Model (for the second tab) ---
        finbert_path = os.path.join(project_root, "finance", "finbert_emotion_model")
        st.info(f"Loading financial model from: {finbert_path}")
        if os.path.exists(finbert_path):
            models['finbert_tokenizer'] = AutoTokenizer.from_pretrained(finbert_path)
            models['finbert_model'] = AutoModelForSequenceClassification.from_pretrained(finbert_path)
        else:
            # Display a warning if the financial model hasn't been trained yet
            st.sidebar.warning("Fine-tuned FinBERT model not found. Please run `python finance/train_finbert.py` from your project root.")
            models['finbert_tokenizer'] = None
            models['finbert_model'] = None

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    return models

# Load all models at startup
models = load_models()


def main():
    st.sidebar.image('https://i.ibb.co/rsbYCsN/senttext-low-resolution-logo-white-on-black-background.png', use_column_width=True)
    
    with st.sidebar:
        choice = option_menu(
            "Menu", ["Emotion Analyzer", "Financial Analysis", "About"],
            icons=['emoji-smile', 'bank', 'info-circle'],
            menu_icon="cast", default_index=0
        )

    if choice == "Emotion Analyzer":
        render_emotion_analyzer()
    elif choice == "Financial Analysis":
        render_financial_analyzer()
    elif choice == "About":
        render_about_page()