import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
import pandas as pd
import re
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Multilingual Summarization Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stTextArea textarea { font-size: 16px; }
.big-font { font-size: 28px !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING (SAFE + FALLBACK)
# ==========================================
@st.cache_resource
def load_model():
    try:
        return pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            device=-1
        )
    except:
        return None

summarizer = load_model()

# ==========================================
# FALLBACK SUMMARIZER (NO ML CRASH OPTION)
# ==========================================
def simple_summary(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:max_sentences])

# ==========================================
# LOAD BAD WORDS CSV
# ==========================================
csv_file = "english_hindi_badwords_500.csv"

@st.cache_data
def load_bad_words():
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            words = (
                df["word"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .tolist()
            )

            return list(set(words))
        return []
    except:
        return []

csv_bad_words = load_bad_words()

default_bad_words = [
    "idiot", "stupid", "damn",
    "moron", "fool", "useless",
    "chutiya", "kamina",
    "बेवकूफ", "हरामी"
]

all_bad_words = list(set(default_bad_words + csv_bad_words))

# ==========================================
# FUNCTIONS
# ==========================================
def contains_hindi(text):
    return any("\u0900" <= c <= "\u097F" for c in text)

def translate_text(text, target):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except:
        return text

def summarize_text(text):
    if len(text.split()) < 30:
        return text

    try:
        if summarizer:
            words = len(text.split())
            return summarizer(
                text,
                max_length=min(120, words),
                min_length=max(20, words // 2),
                do_sample=False
            )[0]["summary_text"]
        else:
            return simple_summary(text)
    except:
        return simple_summary(text)

def detect_bad_words(text):
    text = text.lower()
    found = []

    for word in all_bad_words:
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            found.append(word)

    return list(set(found))

# ==========================================
# UI
# ==========================================
st.markdown('<p class="big-font">🤖 Multilingual Summarization Chatbot</p>', unsafe_allow_html=True)

st.write("Supports English, Hindi, Hinglish + Bad Word Detection")

# CSV status
if csv_bad_words:
    st.success(f"CSV Loaded ({len(csv_bad_words)} words)")
else:
    st.warning("CSV not found (using default list)")

# Input
text = st.text_area("Enter Text:", height=200)

# Output language
output_lang = st.selectbox("Output Language", ["English", "Hindi", "Hinglish"])

# ==========================================
# MAIN BUTTON
# ==========================================
if st.button("Generate Summary 🚀"):

    if not text.strip():
        st.warning("Please enter text")
        st.stop()

    # ALWAYS translate to English first
    working = translate_text(text, "en")

    # Summarize
    with st.spinner("Processing..."):
        summary = summarize_text(working)

    # Output conversion
    if output_lang == "Hindi":
        summary = translate_text(summary, "hi")

    elif output_lang == "Hinglish":
        summary = translate_text(summary, "hi")
        summary = summary.replace("है", "hai").replace("और", "aur")

    # Output
    st.subheader("📌 Summary")
    st.success(summary)

    # Safety check
    st.subheader("🛡 Safety Check")

    bad_words = detect_bad_words(text)

    if bad_words:
        st.error("Unparliamentary language detected")
        st.write(bad_words)
    else:
        st.info("Clean input detected")

# Footer
st.markdown("---")
st.caption("Streamlit + Transformers + NLP Project")
