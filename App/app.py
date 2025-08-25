# app.py

# Core Pkgs
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_option_menu import option_menu
import os # <-- Import the 'os' module

# --- Configuration and Model Loading ---

# Set page configuration
st.set_page_config(
    page_title="SentText - Emotion Analysis",
    page_icon="🙂",
    layout="centered",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model and tokenizer from the local directory.
    Uses st.cache_resource to load the model only once.
    """
    try:
        # --- KEY CHANGE: Build the correct path based on your file structure ---
        # Get the directory of the current script (e.g., .../App/)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the main project root (e.g., .../Sentiment-Analysis-App/)
        project_root = os.path.dirname(app_dir)
        # Now, join the root path with the 'Models' folder and the model name
        model_path = os.path.join(project_root, "Models", "sentiment_model_tiny")
        
        st.info(f"Attempting to load model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Please make sure the folder '{model_path}' exists and contains the model files.")
        return None, None

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
    st.image('https://i.ibb.co/rsbYCsN/senttext-low-resolution-logo-white-on-black-background.png', use_container_width=True)
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
        This application classifies text into granular emotions like **joy, sadness, anger, fear,** and more using a **Tiny BERT** model that has been fine-tuned on a dataset of emotional texts.
        """)

if __name__ == '__main__':
    main()
