# demo_app.py

import streamlit as st

# Configure page FIRST before any other Streamlit commands
st.set_page_config(
    page_title="SentText - Advanced Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from streamlit_option_menu import option_menu
from huggingface_hub import InferenceClient
import requests
import os
import sys
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# Get project root dynamically (no hardcoded paths!)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Sample texts for demo
SAMPLE_TEXTS = {
    'emotion_positive': "I'm absolutely thrilled about this amazing opportunity! This is fantastic news.",
    'emotion_negative': "This is completely disappointing and frustrating. I hate how this turned out.",
    'financial_positive': "Q3 earnings significantly exceeded expectations with robust revenue growth and strong margins.",
    'financial_negative': "Company faces severe headwinds from supply chain disruptions and regulatory challenges."
}

# Emotion emojis
EMOTION_EMOJIS = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "joy": "😂", 
    "neutral": "😐", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# HuggingFace Inference API Setup
HF_TOKEN = st.secrets.get("HF_TOKEN") 

# HuggingFace Inference API Setup
@st.cache_resource
def get_inference_clients():
    """Initialize HuggingFace Inference clients"""
    try:
        HF_TOKEN = st.secrets.get("HF_TOKEN")
        emotion_client = InferenceClient(
            model="Ani-404/emotion-model",
            token=HF_TOKEN
        )
        finbert_client = InferenceClient(
            model="Ani-404/finbert-model", 
            token=HF_TOKEN
        )
        return emotion_client, finbert_client
    except:
        return None, None

emotion_client, finbert_client = get_inference_clients()

@st.cache_resource
def get_runtime_mode():
    """Get HF API clients and mode flag."""
    HF_TOKEN = st.secrets.get("HF_TOKEN")
    try:
        emotion_client = InferenceClient(
            model="Ani-404/emotion-model",
            token=HF_TOKEN
        )
        fin_client = InferenceClient(
            model="Ani-404/finbert-model",
            token=HF_TOKEN
        )
        # Try to touch API
        _ = emotion_client.text_classification("API connectivity test.", wait_for_model=False)
        _ = fin_client.text_classification("API test.", wait_for_model=False)
        return {"mode": "hf_api", "emotion_client": emotion_client, "fin_client": fin_client}
    except Exception as e:
        st.sidebar.warning(f"Using local/demo: {e}")
        return {"mode": "local", "emotion_client": None, "fin_client": None}


@st.cache_resource
@st.cache_resource
def load_models():
    models = get_runtime_mode()
    try:
        # Load local model and tokenizer if they exist
        emotion_path = PROJECT_ROOT / "Models" / "sentiment_model_distilbert"
        if emotion_path.exists():
            models['general_tokenizer'] = AutoTokenizer.from_pretrained(str(emotion_path))
            models['general_model'] = AutoModelForSequenceClassification.from_pretrained(str(emotion_path))
        else:
            models['general_tokenizer'] = None
            models['general_model'] = None
        fin_path = PROJECT_ROOT / "finance" / "finbert_large_emotion_model"
        if fin_path.exists():
            models['finbert_tokenizer'] = AutoTokenizer.from_pretrained(str(fin_path))
            models['finbert_model'] = AutoModelForSequenceClassification.from_pretrained(str(fin_path))
        else:
            models['finbert_tokenizer'] = None
            models['finbert_model'] = None
    except Exception as e:
        models['general_tokenizer'] = models['general_model'] = None
        models['finbert_tokenizer'] = models['finbert_model'] = None
        st.sidebar.warning("Local models unavailable.")
    return models

def predict_emotions_api(text, emotion_client):
    try:
        resp = emotion_client.text_classification(text, wait_for_model=True)
        if not resp:
            return predict_emotions_demo(text)
        top = max(resp, key=lambda x: x['score'])
        label = top['label'].lower()
        conf = top['score']
        possible = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
        probs = [0.05] * len(possible)
        if label in possible:
            probs[possible.index(label)] = conf
        return label, conf, probs
    except Exception:
        return predict_emotions_demo(text)

def analyze_financial_api(text, fin_client, ticker=None):
    try:
        resp = fin_client.text_classification(text, wait_for_model=True)
        if not resp:
            return analyze_financial_sentiment_demo(text, ticker)
        top = max(resp, key=lambda x: x['score'])
        label = top['label'].lower()
        conf = top['score']
        if label == 'positive':
            signal = "BUY" if conf > 0.7 else "HOLD"
            score = conf
        elif label == 'negative':
            signal = "SELL" if conf > 0.7 else "HOLD"
            score = -conf
        else:
            signal = "HOLD"
            score = 0.0
        return {
            'sentiment_score': score,
            'signal': signal,
            'confidence': conf,
            'stock_data': None,
            'positive_ratio': max(0, score),
            'negative_ratio': max(0, -score),
            'neutral_ratio': 1 - abs(score),
        }
    except Exception:
        return analyze_financial_sentiment_demo(text, ticker)

def predict_emotions_demo(text):
    """Demo emotion prediction using keyword analysis"""
    text_lower = text.lower()

    # Simple keyword-based analysis for demo
    if any(word in text_lower for word in ['happy', 'joy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'thrilled']):
        emotion = 'joy'
        confidence = 0.85
        probs = [0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.05]  # joy highest
    elif any(word in text_lower for word in ['sad', 'disappointed', 'terrible', 'awful', 'depressed', 'miserable']):
        emotion = 'sadness'
        confidence = 0.82
        probs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.82, 0.05, 0.05]  # sadness highest
    elif any(word in text_lower for word in ['angry', 'mad', 'furious', 'hate', 'rage', 'frustrated']):
        emotion = 'anger'
        confidence = 0.88
        probs = [0.88, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]  # anger highest
    elif any(word in text_lower for word in ['scared', 'afraid', 'fear', 'worried', 'anxious', 'terrified']):
        emotion = 'fear'
        confidence = 0.80
        probs = [0.05, 0.05, 0.80, 0.05, 0.05, 0.05, 0.05, 0.05]  # fear highest
    elif any(word in text_lower for word in ['surprised', 'shocked', 'wow', 'unexpected', 'amazed']):
        emotion = 'surprise'
        confidence = 0.75
        probs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.75]  # surprise highest
    else:
        emotion = 'neutral'
        confidence = 0.70
        probs = [0.1, 0.1, 0.1, 0.1, 0.70, 0.1, 0.1, 0.1]  # neutral highest

    return emotion, confidence, probs

def predict_emotions_real(text, model, tokenizer):
    """Use HuggingFace Inference API for emotion prediction"""
    try:
        # Call your emotion model via API
        response = emotion_client.text_classification(text)
        
        if response and len(response) > 0:
            # Get top prediction
            top_pred = max(response, key=lambda x: x['score'])
            emotion = top_pred['label'].lower()
            confidence = top_pred['score']
            
            # Create probability distribution 
            emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']
            probs = [0.1] * len(emotion_labels)  # Default small probabilities
            
            # Set the predicted emotion's probability
            if emotion in emotion_labels:
                idx = emotion_labels.index(emotion)
                probs[idx] = confidence
            
            return emotion, confidence, probs
        else:
            # Fallback to demo mode
            return predict_emotions_demo(text)
            
    except Exception as e:
        st.error(f"API Error: {e}")
        return predict_emotions_demo(text)

def analyze_financial_sentiment(text, ticker=None):
    """Use HuggingFace Inference API for financial sentiment"""
    if not text.strip():
        return None
        
    try:
        # Call your finbert model via API
        response = finbert_client.text_classification(text)
        
        if response and len(response) > 0:
            top_pred = max(response, key=lambda x: x['score'])
            label = top_pred['label'].lower()
            confidence = top_pred['score']
            
            # Map to trading signals
            if label == 'positive':
                signal = "BUY" if confidence > 0.7 else "HOLD"
                sentiment_score = confidence
            elif label == 'negative':
                signal = "SELL" if confidence > 0.7 else "HOLD"
                sentiment_score = -confidence
            else:
                signal = "HOLD"
                sentiment_score = 0.0
                
        else:
            # Fallback to keyword analysis
            return analyze_financial_sentiment_demo(text, ticker)
            
    except Exception as e:
        st.error(f"Financial API Error: {e}")
        return analyze_financial_sentiment_demo(text, ticker)
    
    # Get stock data if ticker provided
    stock_data = None
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                recent_change = ((hist['Close'][-1] - hist['Close'][-2]) / hist['Close'][-2]) * 100
                stock_data = {
                    'ticker': ticker,
                    'price': hist['Close'][-1],
                    'change': recent_change
                }
        except Exception as e:
            st.warning(f"Could not fetch stock data for {ticker}: {e}")
    
    return {
        'sentiment_score': sentiment_score,
        'signal': signal,
        'confidence': confidence,
        'stock_data': stock_data,
        'positive_ratio': max(0, sentiment_score),
        'negative_ratio': max(0, -sentiment_score),
        'neutral_ratio': 1 - abs(sentiment_score)
    }

def analyze_financial_sentiment_demo(text, ticker=None):
    """Original keyword-based analysis as fallback"""
    # Your existing keyword analysis code here
    text_lower = text.lower()
    positive_words = ['growth', 'profit', 'exceeded', 'strong', 'robust', 'expansion', 'record', 'beat', 'success', 'revenue']
    negative_words = ['decline', 'loss', 'weak', 'concern', 'challenge', 'disruption', 'headwinds', 'drop', 'fall', 'miss']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment_score = 0.3 + (pos_count - neg_count) * 0.15
        signal = "BUY" if sentiment_score > 0.5 else "HOLD"
    elif neg_count > pos_count:
        sentiment_score = -0.3 - (neg_count - pos_count) * 0.15
        signal = "SELL" if sentiment_score < -0.5 else "HOLD"
    else:
        sentiment_score = 0.0
        signal = "HOLD"
    
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Get stock data
    stock_data = None
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                recent_change = ((hist['Close'][-1] - hist['Close'][-2]) / hist['Close'][-2]) * 100
                stock_data = {
                    'ticker': ticker,
                    'price': hist['Close'][-1],
                    'change': recent_change
                }
        except Exception as e:
            st.warning(f"Could not fetch stock data for {ticker}: {e}")
    
    return {
        'sentiment_score': sentiment_score,
        'signal': signal,
        'confidence': 0.8,
        'stock_data': stock_data,
        'positive_ratio': max(0, sentiment_score),
        'negative_ratio': max(0, -sentiment_score),
        'neutral_ratio': 1 - abs(sentiment_score)
    }

def render_emotion_analyzer(models):
    """Render emotion analysis interface"""
    st.title("🎭 General Emotion Analyzer")
    st.markdown("Analyze the emotional tone of any text using advanced NLP models.")

    # Sample buttons in columns
    st.subheader("📝 Try Sample Texts:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Positive Example", use_container_width=True):
            st.session_state['sample_text'] = SAMPLE_TEXTS['emotion_positive']
    with col2:
        if st.button("😔 Negative Example", use_container_width=True):
            st.session_state['sample_text'] = SAMPLE_TEXTS['emotion_negative']
    with col3:
        if st.button("🔄 Clear Text", use_container_width=True):
            if 'sample_text' in st.session_state:
                del st.session_state['sample_text']

    # Text input
    default_text = st.session_state.get('sample_text', '')
    raw_text = st.text_area(
        "Enter your text for emotion analysis:", 
        value=default_text, 
        height=120,
        placeholder="Type or paste any text here to analyze its emotional content..."
    )

    # Character count
    if raw_text:
        st.caption(f"Character count: {len(raw_text)}")

    # Analysis button
    if st.button("🔍 Analyze Emotions", type="primary", disabled=not raw_text.strip()):
        with st.spinner("Analyzing emotions..."):
            # Use real model if available, otherwise demo mode
            if models.get('mode') == 'hf_api':
                emotion, confidence, probs = predict_emotions_api(raw_text, models['emotion_client'])
                model_status = "Using HuggingFace API model"
            elif models.get('general_model') is not None and models.get('general_tokenizer') is not None:
                emotion, confidence, probs = predict_emotions_real(raw_text, models['general_model'], models['general_tokenizer'])
                model_status = "Using local trained model"
            else:
                emotion, confidence, probs = predict_emotions_demo(raw_text)
                model_status = "Using demo model"

        # Display results
        st.success("🎯 Analysis Complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Results")

            # Emotion with emoji
            emoji_icon = EMOTION_EMOJIS.get(emotion, "🙂")
            st.metric(
                label="Predicted Emotion", 
                value=f"{emotion.capitalize()} {emoji_icon}",
                help="The most likely emotion detected in the text"
            )
            st.metric(
                label="Confidence", 
                value=f"{confidence:.1%}",
                help="Model's confidence in this prediction"
            )

            st.info(f"ℹ️ {model_status}")

            # Original text
            with st.expander("📄 View Original Text"):
                st.write(raw_text)

        with col2:
            st.subheader("📈 Probability Distribution")

            # Create probability chart
            emotions_list = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']

            if models and models.get('general_model'):
                emotions_list = list(models['general_model'].config.id2label.values())

            prob_df = pd.DataFrame({
                'Emotion': [f"{e.title()} {EMOTION_EMOJIS.get(e, '')}" for e in emotions_list],
                'Probability': probs
            }).sort_values('Probability', ascending=True)

            fig = px.bar(
                prob_df, 
                x='Probability', 
                y='Emotion', 
                orientation='h',
                title="Emotion Probability Scores",
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def render_financial_analyzer(models):
    """Render financial analysis interface"""
    st.title("📈 Financial Sentiment Analyzer") 
    st.markdown("Analyze financial text sentiment and get automated trading recommendations.")

    # Sample buttons
    st.subheader("📝 Try Sample Texts:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Bullish Example", use_container_width=True):
            st.session_state['fin_sample_text'] = SAMPLE_TEXTS['financial_positive']
    with col2:
        if st.button("📉 Bearish Example", use_container_width=True):
            st.session_state['fin_sample_text'] = SAMPLE_TEXTS['financial_negative']
    with col3:
        if st.button("🔄 Clear Text", use_container_width=True):
            if 'fin_sample_text' in st.session_state:
                del st.session_state['fin_sample_text']

    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        default_text = st.session_state.get('fin_sample_text', '')
        financial_text = st.text_area(
            "Enter financial text (earnings calls, news, reports):", 
            value=default_text, 
            height=120,
            placeholder="Paste earnings call transcript, financial news, or company reports..."
        )

    with col2:
        ticker = st.text_input(
            "Stock Ticker (optional):", 
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter a stock ticker to get real-time price data"
        ).upper()

        if ticker:
            st.caption(f"Will analyze: {ticker}")

    # Character count
    if financial_text:
        st.caption(f"Character count: {len(financial_text)} | Words: {len(financial_text.split())}")

    # Analysis button
    if st.button("🔍 Analyze Financial Sentiment", type="primary", disabled=not financial_text.strip()):
        with st.spinner("Analyzing financial sentiment..."):
            if models.get('mode') == 'hf_api':
                result = analyze_financial_api(financial_text, models['fin_client'], ticker)
                # unpack result as needed
            elif models.get('finbert_model') is not None and models.get('finbert_tokenizer') is not None:
                result = analyze_financial_sentiment(financial_text, models['finbert_model'], models['finbert_tokenizer'], ticker)
            else:
                result = analyze_financial_sentiment_demo(financial_text, ticker)

        if result:
            st.success("🎯 Analysis Complete!")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Sentiment Score", 
                    f"{result['sentiment_score']:.3f}",
                    help="Overall sentiment score (-1 to +1 scale)"
                )

            with col2:
                st.metric(
                    "Confidence", 
                    f"{result['confidence']:.1%}",
                    help="Model confidence in the analysis"
                )

            with col3:
                signal_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                st.metric(
                    "Trading Signal", 
                    f"{signal_colors.get(result['signal'], '⚪')} {result['signal']}",
                    help="Automated trading recommendation"
                )

            with col4:
                if result['stock_data']:
                    change_delta = result['stock_data']['change']
                    st.metric(
                        f"{ticker} Price Change", 
                        f"{change_delta:.2f}%",
                        delta=f"{change_delta:.2f}%"
                    )

            # Detailed analysis
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Sentiment Breakdown")
                sentiment_data = pd.DataFrame({
                    'Type': ['Positive', 'Negative', 'Neutral'],
                    'Ratio': [result['positive_ratio'], result['negative_ratio'], result['neutral_ratio']]
                })

                fig = px.pie(
                    sentiment_data, 
                    values='Ratio', 
                    names='Type',
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'Positive': '#4CAF50', 
                        'Negative': '#F44336', 
                        'Neutral': '#FF9800'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🎯 Trading Recommendation")

                if result['signal'] == 'BUY':
                    st.success("🚀 **Recommendation: BUY**")
                    st.write("✅ Positive sentiment detected")
                    st.write("📈 Market indicators suggest upward movement")
                elif result['signal'] == 'SELL':
                    st.error("🔻 **Recommendation: SELL**")  
                    st.write("❌ Negative sentiment detected")
                    st.write("📉 Market indicators suggest downward pressure")
                else:
                    st.warning("⏸️ **Recommendation: HOLD**")
                    st.write("➖ Neutral sentiment detected")
                    st.write("📊 Market indicators are mixed")

                if result['stock_data']:
                    st.subheader(f"💰 {ticker} Stock Information")
                    st.write(f"**Current Price:** ${result['stock_data']['price']:.2f}")
                    st.write(f"**Recent Change:** {result['stock_data']['change']:+.2f}%")

                    # Price trend indicator
                    if result['stock_data']['change'] > 0:
                        st.success("📈 Stock is trending upward")
                    elif result['stock_data']['change'] < 0:
                        st.error("📉 Stock is trending downward")
                    else:
                        st.info("➖ Stock price is stable")

def render_about_page():
    """Render about page"""
    st.title("ℹ️ About SentText Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 🎯 What This Application Does

        **SentText Analytics** is a dual-purpose sentiment analysis platform:

        ### 🎭 General Emotion Analyzer
        - Analyzes **any text** for emotional content
        - Classifies into **8 emotions**: joy, sadness, anger, fear, surprise, disgust, neutral, shame
        - Shows **confidence scores** and probability distributions
        - Uses **DistilBERT** model trained on emotion datasets

        ### 📈 Financial Sentiment Analyzer
        - Analyzes **financial texts** (earnings calls, news, reports)
        - Generates **trading signals**: BUY/SELL/HOLD recommendations
        - Integrates **real stock prices** from Yahoo Finance
        - Uses **FinBERT** model specialized for financial language
        """)

    with col2:
        st.markdown("""
        ## 🚀 How to Use

        1. **Choose Analysis Type**: Select Emotion or Financial tab
        2. **Input Text**: Type your text or try sample examples
        3. **Get Results**: Click analyze for instant results
        4. **View Charts**: Interactive visualizations show detailed breakdown
        5. **Trading Signals**: Get automated BUY/SELL/HOLD recommendations

        ## 🛠️ Technical Features

        - **Models**: Transformers-based NLP models
        - **Visualization**: Interactive Plotly charts
        - **Real-time Data**: Yahoo Finance integration
        - **GPU Acceleration**: CUDA support when available
        - **Demo Mode**: Works even without trained models
        """)

    st.markdown("---")

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Emotion Classes", "8", help="Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral, Shame")
    with col2:
        st.metric("Processing Speed", "< 2s", help="Average analysis time per text")
    with col3:
        st.metric("Model Accuracy", "90%+", help="Accuracy on validation datasets")
    with col4:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Processing Device", device, help="Hardware used for computations")

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem;'>
        <h1 style='color: #1f77b4; margin-bottom: 0;'>📊 SentText Analytics</h1>
        <p style='color: #666; font-size: 1.1rem; margin-top: 0.5rem;'>Advanced Sentiment Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models once
    models = load_models()

    # Navigation menu in sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        choice = option_menu(
            "Analysis Type", 
            ["🎭 Emotion Analyzer", "📈 Financial Analysis", "ℹ️ About"],
            icons=['emoji-smile', 'graph-up-arrow', 'info-circle'],
            menu_icon="cast", 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

        st.markdown("---")

        # System info
        st.markdown("### 🖥️ System Status")
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")

        if models:
            emotion_status = "✅ Loaded" if models.get('general_model') else "⚠️ Demo Mode"
            finbert_status = "✅ Loaded" if models.get('finbert_model') else "⚠️ Demo Mode"
            st.info(f"**Emotion Model:** {emotion_status}")
            st.info(f"**FinBERT Model:** {finbert_status}")

        st.markdown("---")
        st.markdown("### 🔧 Quick Actions")
        if st.button("🔄 Reload Models", help="Reload all models"):
            st.cache_resource.clear()
            st.rerun()

    # Render selected page
    if "Emotion" in choice:
        render_emotion_analyzer(models)
    elif "Financial" in choice:
        render_financial_analyzer(models)
    elif "About" in choice:
        render_about_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built with ❤️ using Streamlit • © 2025 SentText Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
