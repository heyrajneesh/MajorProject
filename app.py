import streamlit as st
import streamlit.components.v1 as components
import pickle

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from pathlib import Path
import streamlit_authenticator as stauth
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import numpy as np
import speech_recognition as sr
import pandas as pd

# 🔹 Page config
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="💬",
    layout="centered"
)

# 🔹 Load model & tokenizer
model = load_model("model2.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

# 🔹 Labels
labels = {
    0: "🔴 Hate Speech",
    1: "🟠 Offensive Language",
    2: "🟢 Normal"
}

# 🔹 Custom CSS (for styling)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #bbbbbb;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# 🔹 Title
st.markdown('<div class="title">💬 Hate Speech Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered text classification using RNN</div>', unsafe_allow_html=True)

st.write("")

# 🔹 Input box
user_input = st.text_area("✍️ Enter your text below:", height=150)

# 🔹 Predict button
if st.button("🚀 Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len)

        # Prediction
        pred = model.predict(padded)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)

        # 🔹 Result section
        st.write("---")
        st.subheader("📊 Prediction Result")

        st.success(f"{labels[pred_class]}")
        st.info(f"Confidence: {confidence*100:.2f}%")

# 🔹 Footer
st.write("---")
st.caption("Built using TensorFlow & Streamlit | NLP Project")
