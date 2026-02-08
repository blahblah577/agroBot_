import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import google.genai as genai
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AgroBot 🌿",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(to bottom, #e8f5e9, #c8e6c9);
}
.title {
    text-align: center;
    color: #1b5e20;
    font-size: 48px;
}
.subtitle {
    text-align: center;
    color: #388e3c;
    font-size: 18px;
}
.stChatInput textarea {
    border-radius: 30px;
    border: 2px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)


# ---------------- API ----------------
from google import genai

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = "models/gemini-1.5-pro"


# ---------------- LOAD MODELS ----------------
cnn_model = tf.keras.models.load_model("cnn_model.h5")
nlp_model = pickle.load(open("nlp_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

CLASS_NAMES = ['curl', 'healthy', 'slug', 'spot'];

# ---------------- FUNCTIONS ----------------
def predict_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = cnn_model.predict(img)
    return CLASS_NAMES[np.argmax(pred)]

def generate_ai_response(disease, user_text):
    prompt = f"""
You are AgroBot 🌱, an expert plant disease assistant.

Detected disease: {disease}
User description: {user_text}

Explain:
- Disease overview
- Symptoms
- Causes
- Treatment (organic + chemical)
- Prevention tips

Be friendly, farmer-friendly, and vary wording every time.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text

    except Exception as e:
        return f"""
🌿 **{disease} Detected**

This plant disease affects growth and yield.

Basic guidance:
• Remove infected leaves
• Improve airflow
• Avoid overwatering
• Use appropriate treatment

(API fallback used)
"""

# ---------------- UI ----------------
st.markdown("<h1 class='title'>AgroBot 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Plant Disease Diagnosis & Chatbot</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📷 Upload Plant Image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        disease = predict_image(image)
        st.success(f"Detected: {disease}")

with col2:
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Describe symptoms or ask for help 🌿")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        disease_context = disease if uploaded_file else "Plant Disease"
        reply = generate_ai_response(disease_context, user_input)

        st.chat_message("assistant").markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
