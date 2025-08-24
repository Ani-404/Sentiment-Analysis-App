# app.py

# Core Pkgs
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_option_menu import option_menu

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
    Loads the fine-tuned BERT model and tokenizer from the local directory.
    Uses st.cache_resource to load the model only once.
    """
    try:
        model_path = "./sentiment_model_bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error("Please make sure you have run the 'emotional_model.py' script to train and save the model in the 'sentiment_model_bert' directory.")
        return None, None

# Load the model and tokenizer
tokenizer, model = load_model()

# --- UI and Prediction Functions ---

# Dictionary for mapping emotions to emojis
# Ensure these labels match EXACTLY what's in your dataset
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨",
    "joy": "😂",
    "neutral": "😐",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

def predict_emotions(docx):
    """
    Predicts the emotion of a single text string using the loaded BERT model.
    
    Args:
        docx (str): The input text.
        
    Returns:
        str: The predicted emotion label.
    """
    if model is None or tokenizer is None:
        return "Model not loaded"
        
    # Tokenize the input text
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Perform prediction
    with torch.no_grad():
        logits = model(**inputs).logits
        
    # Get the predicted class ID and map it to the label
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return model.config.id2label[predicted_class_id]

def get_prediction_proba(docx):
    """
    Gets the prediction probabilities for each emotion for a single text string.
    
    Args:
        docx (str): The input text.
        
    Returns:
        np.ndarray: An array of probabilities for each class.
    """
    if model is None or tokenizer is None:
        return np.array([])

    # Tokenize the input text
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Perform prediction and apply softmax to get probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    return probabilities.numpy()

# --- Main Application ---

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.image('https://i.ibb.co/rsbYCsN/senttext-low-resolution-logo-white-on-black-background.png', use_column_width=True)
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

                    # Get prediction and probabilities
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

                        # Create and display the probability chart
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
        Sentiment Analysis is a common text classification tool that analyzes an incoming message and tells whether the underlying sentiment is positive, negative, or neutral. 
        
        This application goes a step further by classifying the text into more granular emotions like **joy, sadness, anger, fear,** and more.
        
        ### How it Works
        The app uses a **DistilBERT** model that has been fine-tuned on a dataset of emotional texts. When you input a sentence, the model processes it and outputs the most likely emotion along with a confidence score.
        
        You can input a sentence of your choice and gauge the underlying sentiment by playing with the demo on the **Home** page.
        """)

if __name__ == '__main__':
    main()
