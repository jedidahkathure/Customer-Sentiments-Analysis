import streamlit as st
import requests

st.title("📱 Telco Sentiment Predictor")
user_input = st.text_area("Enter app review:")
if st.button("Predict Sentiment"):
    response = requests.post(
        "https://customer-sentiments-analysis.onrender.com/predict",
        json={"text": user_input}
    )
    result = response.json()
    st.write(f"**Sentiment:** {result['sentiment']}")
    st.write(f"**Confidence:** {result['confidence']}")
