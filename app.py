import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import google.genai as genai
import os
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgroBot 🌿",
    page_icon="🌱",
    layout="centered"
)

# ---------------- CSS ----------------
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

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = [{
        "role": "assistant",
        "content": (
            "👋 Hi! I’m **AgroBot 🌱**.\n\n"
            "I specialize **only in plant health**, diseases, and care.\n"
            "Upload a plant image or describe symptoms to get started."
        )
    }]

if "detected_disease" not in st.session_state:
    st.session_state.detected_disease = "Plant Disease"

# ---------------- API ----------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
MODEL_NAME = "models/gemini-1.5-pro"

# ---------------- LOAD MODELS ----------------
cnn_model = tf.keras.models.load_model("cnn_model.h5")
nlp_model = pickle.load(open("nlp_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

CLASS_NAMES = ['curl', 'healthy', 'slug', 'spot']

# ---------------- UTIL FUNCTIONS ----------------
def is_plant_related(text):
    keywords = [
        "plant", "leaf", "leaves", "crop", "soil", "fungus", "disease",
        "fertilizer", "pesticide", "watering", "root", "stem"
    ]
    return any(word in text.lower() for word in keywords)

def predict_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = cnn_model.predict(img)
    return CLASS_NAMES[np.argmax(pred)]

def generate_ai_response(disease, user_text):
    if not is_plant_related(user_text):
        return (
            "🌱 I’m designed to help **only with plant-related questions**.\n\n"
            "Please ask about plant diseases, symptoms, treatment, or care."
        )

    prompt = f"""
You are AgroBot 🌱, a professional AI assistant for plant health.

Avoid repeating phrasing.
Do NOT answer non-plant questions.

Context:
Detected disease: {disease}
User message: {user_text}

Respond naturally with:
- Brief explanation
- Symptoms
- Treatment (organic + chemical)
- Prevention tips
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text.strip()

    except Exception:
        return (
            f"🌿 Based on available information, **{disease}** may affect plant growth.\n\n"
            "### Care Tips\n"
            "- Remove infected parts\n"
            "- Ensure airflow and sunlight\n"
            "- Avoid excess moisture\n"
            "- Use suitable plant-safe treatment\n\n"
            "_AI service temporarily unavailable_"
        )

# ---------------- UI HEADER ----------------
st.markdown("<h1 class='title'>AgroBot 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Plant Disease Diagnosis & Chatbot</p>", unsafe_allow_html=True)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📷 Upload a plant image (optional)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded plant image", use_container_width=True)

    detected = predict_image(image)
    st.session_state.detected_disease = detected

    st.session_state.chat.append({
        "role": "assistant",
        "content": f"🌿 I analyzed the image and detected **{detected}**.\nYou may ask for treatment or prevention advice."
    })

    st.rerun()

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT (BOTTOM FIXED) ----------------
user_input = st.chat_input("Ask about plant health 🌿")

if user_input:
    st.session_state.chat.append(
        {"role": "user", "content": user_input}
    )

    reply = generate_ai_response(
        st.session_state.detected_disease,
        user_input
    )

    st.session_state.chat.append(
        {"role": "assistant", "content": reply}
    )

    st.rerun()

