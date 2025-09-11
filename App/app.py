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

# Load the model and tokenizer
tokenizer, model = load_model()

# --- UI and Prediction Functions ---

# Dictionary for mapping emotions to emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "joy": "😂",
    "neutral": "😐", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    """
    Predicts the emotion of a single text string using the loaded BERT model.
    """
    if model is None or tokenizer is None:
        return "Model not loaded"
        
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return model.config.id2label[predicted_class_id]

def get_prediction_proba(docx):
    """
    Gets the prediction probabilities for each emotion for a single text string.
    """
    if model is None or tokenizer is None:
        return np.array([])

    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities.numpy()

# --- Main Application ---

def main():
    """
    The main function that runs the Streamlit application.
    """
    # --- KEY CHANGE: Reverted to the older 'use_column_width' argument ---
    st.image('https://i.ibb.co/rsbYCsN/senttext-low-resolution-logo-white-on-black-background.png', use_column_width=True)
    # --- End of Change ---
    
    st.title("SentText Emotion Analyzer")
    st.subheader("Analyze the emotional tone of your text with a fine-tuned BERT model.")

    with st.sidebar:
        choice = option_menu(
            "Menu", ["Home", "About"],
            icons=['house', 'info-circle'],
            menu_icon="cast",
            default_index=0
        )

    if choice == "Home":
        if model is not None:
            with st.form(key='emotion_clf_form'):
                raw_text = st.text_area("Type your text here...", height=150)
                submit_text = st.form_submit_button(label='Analyze')

            if submit_text:
                if raw_text.strip() == "":
                    st.warning("Please enter some text to analyze.")
                else:
                    col1, col2 = st.columns(2)

                    prediction = predict_emotions(raw_text)
                    probability = get_prediction_proba(raw_text)

                    with col1:
                        st.success("Original Text")
                        st.write(raw_text)

                        st.success("Prediction Probability")
                        labels = list(model.config.id2label.values())
                        proba_df = pd.DataFrame(probability, columns=labels)
                        st.write(proba_df.T.rename(columns={0: 'Probability'}))

                    with col2:
                        st.success("Prediction")
                        emoji_icon = emotions_emoji_dict.get(prediction, "🙂")
                        st.metric(label="Predicted Emotion", value=f"{prediction.capitalize()} {emoji_icon}")
                        st.metric(label="Confidence", value=f"{np.max(probability):.4f}")
                        
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]
                        
                        fig = alt.Chart(proba_df_clean).mark_bar().encode(
                            x=alt.X('emotions', sort=None),
                            y='probability',
                            color='emotions',
                            tooltip=['emotions', 'probability']
                        ).properties(
                            title="Emotion Probabilities"
                        )
                        st.altair_chart(fig, use_container_width=True)
        else:
            st.info("The model is not available. Please check the error messages above.")

    else: # About page
        st.header("About SentText")
        st.markdown("""
        This application provides nuanced emotional analysis that goes beyond simple positive or negative sentiment. By classifying text into granular categories such as joy, anger, sadness, and fear - SentText offers a deeper understanding of the underlying tone and intent of a message.

        The analysis is powered by a DistilBERT model, a streamlined and efficient version of the powerful BERT language model. This model has been specifically fine-tuned on a large dataset of emotional texts to accurately identify and differentiate between various human emotions. The result is a tool that provides fast, reliable, and insightful analysis of textual communication.
        """)

if __name__ == '__main__':
    main()
