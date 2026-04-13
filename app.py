# app.py
import os
import warnings

# ---------------------------------------------------
# Hugging Face Cache Configuration (For Render)
# ---------------------------------------------------
# Hugging Face Cache Configuration
import os
import warnings

os.environ["HF_HOME"] = "/opt/render/project/src/.hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
import streamlit as st
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector with AI Summary")
st.write(
    "Detect whether a news article is **Real or Fake** with explanations, "
    "AI-generated summaries, and credible alternatives."
)

# ---------------------------------------------------
# Disclaimer
# ---------------------------------------------------
st.warning("""
⚠️ **Disclaimer:**  
This Fake News Detector is an AI-based prototype developed for educational and research purposes.  
Predictions and summaries may not always be accurate. Always verify information using trusted
sources such as Reuters, BBC, WHO, and official government websites.
""")

# ---------------------------------------------------
# Load Fake News Detection Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ---------------------------------------------------
# Load AI Summarizer
# ---------------------------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # CPU mode for Render
    )

summarizer = load_summarizer()

# ---------------------------------------------------
# Extract Text from URL
# ---------------------------------------------------
def extract_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])
    except Exception:
        return ""

# ---------------------------------------------------
# AI-Based Summarization
# ---------------------------------------------------
def summarize_text(text):
    try:
        text = text.strip().replace("\n", " ")
        if len(text.split()) < 40:
            return text  # Return original text if too short

        summary = summarizer(
            text[:1024],
            max_length=80,
            min_length=25,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Summary could not be generated: {e}"

# ---------------------------------------------------
# Generate Reason for Prediction
# ---------------------------------------------------
def generate_reason(text, prediction, confidence):
    feature_names = np.array(vectorizer.get_feature_names_out())
    X_vec = vectorizer.transform([text]).toarray()[0]

    influential_words = []
    if hasattr(model, "feature_log_prob_"):
        contribution = X_vec * (
            model.feature_log_prob_[1] - model.feature_log_prob_[0]
        )
        top_indices = np.argsort(np.abs(contribution))[-5:][::-1]
        influential_words = [
            feature_names[i] for i in top_indices if X_vec[i] > 0
        ]

    sensational_keywords = [
        "miracle", "shocking", "unbelievable", "secret",
        "hoax", "rumor", "aliens", "instant", "click",
        "earn", "cure"
    ]

    detected = [w for w in sensational_keywords if w in text.lower()]

    if prediction == 1:
        reason = (
            f"This news is classified as REAL with {confidence:.2f}% confidence. "
            f"Key informative terms include: {', '.join(influential_words)}."
        )
        if not detected:
            reason += " The language appears factual and non-sensational."
    else:
        reason = (
            f"This news is classified as FAKE with {confidence:.2f}% confidence. "
            f"The prediction is influenced by suspicious terms such as: "
            f"{', '.join(influential_words)}."
        )
        if detected:
            reason += f" It also contains sensational keywords like: {', '.join(detected)}."

    return reason

# ---------------------------------------------------
# Suggest Real News for Fake Inputs
# ---------------------------------------------------
def suggest_real_news(text, prediction):
    if prediction == 1:
        return "This news appears credible. Please verify it with trusted sources."

    text_lower = text.lower()
    patterns = {
        "aliens": "There is no scientific evidence confirming extraterrestrial visits. Agencies like NASA continue researching life beyond Earth.",
        "miracle cure": "There is currently no instant cure for cancer. Treatments such as chemotherapy and immunotherapy are scientifically validated.",
        "earn money": "Financial experts warn against get-rich-quick schemes.",
        "click here": "Cybersecurity experts caution users against suspicious links that may lead to scams.",
        "time travel": "Time travel remains theoretical and has not been achieved in practice.",
        "immortality": "Modern medicine improves lifespan, but immortality is not scientifically possible."
    }

    for keyword, correction in patterns.items():
        if keyword in text_lower:
            return correction

    return (
        "No verified evidence supports this claim. Refer to trusted sources such as "
        "Reuters, BBC, or official government websites."
    )

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
user_input = st.text_area("✍️ Enter News Text", height=200)
url = st.text_input("🌐 OR Paste News URL")

if url:
    extracted_text = extract_text(url)
    if extracted_text:
        st.success("✅ Text extracted from URL")
        user_input = extracted_text[:2000]
        st.text_area("Extracted Content", user_input, height=200)
    else:
        st.error("❌ Failed to fetch content from URL")

# ---------------------------------------------------
# Analyze Button
# ---------------------------------------------------
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter news text.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0]
        confidence = max(probability) * 100

        st.subheader("🧾 Prediction Result")
        if prediction == 1:
            st.success(f"✅ REAL NEWS ({confidence:.2f}%)")
        else:
            st.error(f"❌ FAKE NEWS ({confidence:.2f}%)")

        st.subheader("📝 News Summary")
        st.write(summarize_text(user_input))

        st.subheader("📊 Reason for Prediction")
        st.info(generate_reason(user_input, prediction, confidence))

        st.subheader("📰 If Fake, What Could Be the Real News?")
        st.write(suggest_real_news(user_input, prediction))

        st.subheader("🔑 Important Words Influencing the Model")
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = X_input.toarray()[0]
        top_indices = scores.argsort()[-10:][::-1]

        for i in top_indices:
            if scores[i] > 0:
                st.write(f"👉 {feature_names[i]} ({scores[i]:.3f})")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("Developed using Machine Learning, NLP, Hugging Face Transformers, and Streamlit.")