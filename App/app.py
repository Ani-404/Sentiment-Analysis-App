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
import sys

print('Starting App...')

PROJECT_ROOT = r"C:\Users\anime\OneDrive\Desktop\Sentiment-Analysis-App"
# This block adds your project folder to Python's search path so it can find 'finance'.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import New Finance Modules ---
# This will now work correctly because of the path correction above.
from finance.processor import (
    get_huggingface_sentiment,
    get_twitter_credentials,
    get_twitter_data,
)
from finance.modeling_analysis import run_prediction_model


# Configuration and Model Loading

# Set page configuration
st.set_page_config(
    page_title="SentText - Advanced Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Loads both models using the hardcoded project root path."""
    models = {}
    try:
        # --- Load General Emotion Model ---
        # This path is for the "Emotion Analyzer" tab.
        # It should point to your trained DistilBERT model.
        general_model_path = os.path.join(PROJECT_ROOT, "Models", "sentiment_model_distilbert")
        st.info(f"Loading general model from: {general_model_path}")
        models['general_tokenizer'] = AutoTokenizer.from_pretrained(general_model_path)
        models['general_model'] = AutoModelForSequenceClassification.from_pretrained(general_model_path)
        
        # --- Load Fine-Tuned Financial Model ---
        # This path is for the "Financial Analysis" tab.
        # It should point to your final, high-accuracy model (e.g., finbert_large_emotion_model).
        finbert_path = os.path.join(PROJECT_ROOT, "finance", "finbert_large_emotion_model") 
        st.info(f"Loading financial model from: {finbert_path}")
        if os.path.exists(finbert_path):
            models['finbert_tokenizer'] = AutoTokenizer.from_pretrained(finbert_path)
            models['finbert_model'] = AutoModelForSequenceClassification.from_pretrained(finbert_path)
        else:
            # This warning will appear if the folder is not found
            st.sidebar.warning("Fine-tuned FinBERT model not found. Please run the training script.")
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

# Page 1: Original Emotion Analyzer

def render_emotion_analyzer():
    st.title("General Emotion Analyzer")
    st.subheader("Analyze the emotional tone of any text with a fine-tuned DistilBERT model.")

    if models and models.get('general_model'):
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type your text here...", height=150)
            submit_text = st.form_submit_button(label='Analyze')

        if submit_text and raw_text.strip():
            # This section uses the GENERAL model
            prediction = predict_general_emotions(raw_text, models['general_model'], models['general_tokenizer'])
            probability = get_general_prediction_proba(raw_text, models['general_model'], models['general_tokenizer'])
            
            # (UI code for displaying results - adapted from your original app)
            col1, col2 = st.columns(2)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction Probability")
                labels = list(models['general_model'].config.id2label.values())
                proba_df = pd.DataFrame(probability, columns=labels)
                st.write(proba_df.T.rename(columns={0: 'Probability'}))
            with col2:
                st.success("Prediction")
                emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨", "joy": "😂", "neutral": "😐", "sadness": "😔", "shame": "😳", "surprise": "😮"}
                emoji_icon = emotions_emoji_dict.get(prediction, "🙂")
                st.metric(label="Predicted Emotion", value=f"{prediction.capitalize()} {emoji_icon}")
                st.metric(label="Confidence", value=f"{np.max(probability):.4f}")

    else:
        st.error("General emotion model not loaded. Check model path and files.")

# Helper functions for the general model
def predict_general_emotions(docx, model, tokenizer):
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return model.config.id2label[pred_id]

def get_general_prediction_proba(docx, model, tokenizer):
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.nn.functional.softmax(logits, dim=1).numpy()

# Page 2: New Financial Analyzer 

@st.cache_data
def run_full_finance_pipeline():
    """Caches the full financial data processing and modeling pipeline."""
    if not models or not models.get('finbert_model'):
        return None, None, None, "Fine-tuned FinBERT model not available. Please run the training script."

    finbert_model = models['finbert_model']
    finbert_tokenizer = models['finbert_tokenizer']
    
    df_transcripts = ingest_transcripts(PROJECT_ROOT)
    all_features = []

    for _, row in df_transcripts.iterrows():
        remarks_sentences = preprocess_and_split(row['prepared_remarks'])
        remarks_vectors = get_sentiment_vectors(remarks_sentences, finbert_model, finbert_tokenizer)
        remarks_features = aggregate_vectors_to_features(remarks_vectors, finbert_model, prefix='remarks_')

        qa_sentences = preprocess_and_split(row['analyst_qa'])
        qa_vectors = get_sentiment_vectors(qa_sentences, finbert_model, finbert_tokenizer)
        qa_features = aggregate_vectors_to_features(qa_vectors, finbert_model, prefix='qa_')
        
        combined_features = {**remarks_features, **qa_features}
        returns = get_stock_returns(row['ticker'], row['earnings_date'])
        
        if returns:
            full_feature_row = {**combined_features, **returns, 'ticker': row['ticker'], 'company_name': row['company_name']}
            all_features.append(full_feature_row)

    if not all_features:
        return df_transcripts, None, None, "Could not process transcripts or fetch stock data."

    features_df = pd.DataFrame(all_features)
    model_results, message = run_prediction_model(features_df.dropna())
    return df_transcripts, features_df.dropna(), model_results, message


def render_financial_analyzer():
    st.title("Advanced Earnings Call Analysis")
    st.subheader("Predicting Stock Returns with FinBERT-powered Sentiment Analysis")

    df_transcripts, features_df, model_results, message = run_full_finance_pipeline()

    if features_df is None:
        st.error(message)
        return

    st.header("Company Analysis Dashboard")
    company = st.selectbox("Select Company", options=df_transcripts['company_name'].unique())
    
    selected_features = features_df[features_df['company_name'] == company].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("1-Day Post-Earnings Return", f"{selected_features['return_1d']:.2%}")
    col2.metric("5-Day Post-Earnings Return", f"{selected_features['return_5d']:.2%}")
    if model_results:
        col3.metric("Model MSE on Test Set", f"{model_results['mse']:.6f}")


    st.subheader("Sentiment Analysis Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Prepared Remarks Sentiment")
        remarks_cols = {k.replace('remarks_', '').replace('_mean','').capitalize():v for k,v in selected_features.items() if 'remarks' in k}
        remarks_df = pd.DataFrame.from_dict(remarks_cols, orient='index', columns=['Mean Score'])
        st.bar_chart(remarks_df)

    with col2:
        st.markdown("##### Analyst Q&A Sentiment")
        qa_cols = {k.replace('qa_', '').replace('_mean','').capitalize():v for k,v in selected_features.items() if 'qa_' in k}
        qa_df = pd.DataFrame.from_dict(qa_cols, orient='index', columns=['Mean Score'])
        st.bar_chart(qa_df)

    if model_results:
        st.header("Predictive Model Performance (XGBoost)")
        st.info(message)
        st.subheader("Top 10 Most Predictive Features")
        st.bar_chart(model_results['feature_importance'].head(10))

# Page 3: About Page 

def render_about_page():
    st.header("About SentText")
    st.markdown("""
    This application contains two distinct analysis tools.

    **Emotion Analyzer:** A general-purpose tool that classifies any text into one of eight granular emotions. It is powered by a fine-tuned DistilBERT model, ideal for understanding the emotional tone of reviews, social media posts, or personal messages.

    **Financial Analysis:** A specialized tool designed to analyze the sentiment of corporate earnings calls and test its correlation with stock price performance. It uses a FinBERT model, fine-tuned for emotion analysis, to extract nuanced sentiment features from both the prepared remarks and the analyst Q&A sections of a transcript. An XGBoost model then attempts to predict post-earnings stock returns based on these advanced linguistic features.
    """)

if __name__ == '__main__':
    main()

print("App loaded successfully.")