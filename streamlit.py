# app.py
import streamlit as st
import requests

st.title("Message Classifier")
st.write("Enter a message to predict its tone:")

user_input = st.text_area("Message Text")

if st.button("Predict"):
    if user_input.strip():
        response = requests.post(
            "http://localhost:8000/predict/",
            json={"text": user_input}
        )
        if response.status_code == 200:
            label = response.json()["prediction"]
            st.success(f"Prediction: **{label.upper()}**")
        else:
            st.error("Something went wrong with the prediction.")
    else:
        st.warning("Please enter a message.")
