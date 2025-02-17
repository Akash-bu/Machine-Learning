import streamlit as st
import joblib

vect = joblib.load("vect.pkl")

st.title("Real fake news classifier")
text_model = joblib.load("model.pkl")

input = st.text_input("Enter your news")
output = text_model.predict(vect.transform([input]))
if st.button("Predict"):
    st.title(output[0])
