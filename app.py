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
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stTextArea textarea {
    font-size: 16px;
}
.big-font {
    font-size: 28px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    summarizer = pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6",
        tokenizer="sshleifer/distilbart-cnn-12-6"
    )
    return summarizer

summarizer = load_model()
# ==========================================
# LOAD CSV BAD WORDS
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
        else:
            return []
    except:
        return []

csv_bad_words = load_bad_words()

# ==========================================
# DEFAULT BAD WORDS
# ==========================================
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
        return GoogleTranslator(
            source="auto",
            target=target
        ).translate(text)
    except:
        return text

def summarize_text(text):
    words = len(text.split())

    if words < 30:
        return text

    max_len = min(120, words)
    min_len = max(20, words // 2)

    try:
        result = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )

        return result[0]["summary_text"]

    except:
        return text

def detect_bad_words(text):
    found = []
    text = text.lower()

    for word in all_bad_words:
        pattern = r"\b" + re.escape(word) + r"\b"

        if re.search(pattern, text):
            found.append(word)

    return list(set(found))

# ==========================================
# UI
# ==========================================
st.markdown(
    '<p class="big-font">🤖 Multilingual Summarization Chatbot</p>',
    unsafe_allow_html=True
)

st.write(
    "Supports **English, Hindi, Hinglish**  \n"
    "Includes **Bad Word Detection** using CSV dataset."
)

# CSV Status
if csv_bad_words:
    st.success(f"CSV Loaded Successfully ({len(csv_bad_words)} words)")
else:
    st.warning("CSV file not found. Keep english_hindi_badwords_500.csv in same folder.")

# Input
text = st.text_area(
    "Enter Text Here:",
    height=220,
    placeholder="Type paragraph here..."
)

# Language
output_lang = st.selectbox(
    "Choose Output Language",
    ["English", "Hindi", "Hinglish"]
)

# ==========================================
# BUTTON
# ==========================================
if st.button("Generate Summary 🚀"):

    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    working = text

    # Hindi/Hinglish → English
    if contains_hindi(text):
        working = translate_text(text, "en")

    else:
        working = translate_text(text, "en")

    # Summarize
    with st.spinner("Generating Summary..."):
        summary = summarize_text(working)

    # Convert Output Language
    if output_lang == "Hindi":
        summary = translate_text(summary, "hi")

    elif output_lang == "Hinglish":
        summary = translate_text(summary, "hi")

        summary = (
            summary
            .replace("है", "hai")
            .replace("और", "aur")
            .replace("का", "ka")
            .replace("हैं", "hain")
        )

    # Output
    st.subheader("📌 Summary")
    st.success(summary)

    # Safety Check
    st.subheader("🛡 Language Safety Check")

    bad_words = detect_bad_words(text)

    if bad_words:
        st.error("⚠ Unparliamentary Language Detected")
        st.write(bad_words)
    else:
        st.info("No unparliamentary language detected.")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit + Transformers + NLP")
