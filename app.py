import os
import warnings
import streamlit as st
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# -----------------------------
# Environment Safety (Render)
# -----------------------------
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector with AI Summary")

# -----------------------------
# Load ML Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# Load Summarizer (Lazy)
# -----------------------------
@st.cache_resource
def get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

summarizer = get_summarizer()

# -----------------------------
# Extract text from URL
# -----------------------------
def extract_text(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])
    except:
        return ""

# -----------------------------
# Summarization
# -----------------------------
def summarize(text):
    try:
        text = text.strip()
        if len(text.split()) < 40:
            return text

        out = summarizer(text[:1000], max_length=80, min_length=25, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return f"Summary failed: {str(e)}"

# -----------------------------
# Prediction Reasoning
# -----------------------------
def explain(text, pred, conf):
    features = np.array(vectorizer.get_feature_names_out())
    X = vectorizer.transform([text]).toarray()[0]

    top_words = []
    if hasattr(model, "feature_log_prob_"):
        impact = X * (model.feature_log_prob_[1] - model.feature_log_prob_[0])
        idx = np.argsort(np.abs(impact))[-5:]
        top_words = [features[i] for i in idx if X[i] > 0]

    label = "REAL" if pred == 1 else "FAKE"

    return f"{label} prediction with {conf:.2f}% confidence. Key terms: {', '.join(top_words)}"

# -----------------------------
# UI Inputs
# -----------------------------
text = st.text_area("Enter News Text")
url = st.text_input("OR Enter URL")

if url:
    extracted = extract_text(url)
    if extracted:
        text = extracted[:2000]
        st.success("Text extracted from URL")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Analyze"):
    if not text.strip():
        st.warning("Enter text or URL")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        conf = max(prob) * 100

        st.subheader("Result")
        st.success(f"REAL ({conf:.2f}%)" if pred == 1 else f"FAKE ({conf:.2f}%)")

        st.subheader("Summary")
        st.write(summarize(text))

        st.subheader("Explanation")
        st.info(explain(text, pred, conf))
