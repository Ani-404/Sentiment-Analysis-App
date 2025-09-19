import streamlit as st
from finance.quick_predict import predict_signal

st.title("📈 Quick Stock Signal Demo")
ticker = st.text_input("Ticker", value="AAPL")
news   = st.text_area("Latest headline or paragraph")

if st.button("Get signal") and news.strip():
    signal = predict_signal(ticker.upper(), news)
    emoji  = {"Buy":"🟢", "Hold":"🟡", "Sell":"🔴"}[signal]
    st.markdown(f"### {emoji} **{signal}**")