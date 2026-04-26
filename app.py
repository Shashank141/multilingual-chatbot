import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
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
    font-size:22px !important;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS (DEPLOY FRIENDLY)
# ==========================================
@st.cache_resource
def load_models():
    translator = Translator()

    # Smaller model for cloud deployment
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return translator, summarizer


translator, summarizer = load_models()

# ==========================================
# LOAD CSV BAD WORDS
# ==========================================
csv_file = "english_hindi_badwords_500.csv"

@st.cache_data
def load_bad_words():
    try:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, encoding="utf-8")

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
# KEYWORDS
# ==========================================
hinglish_keywords = [
    "kal", "hai", "mera", "tum", "hum",
    "bhai", "acha", "kya", "nahi",
    "haan", "kamina", "chutiya"
]

# ==========================================
# BACKUP WORDS
# ==========================================
english_bad = [
    "idiot", "stupid", "damn",
    "fool", "moron", "useless"
]

hindi_bad = [
    "बेवकूफ", "हरामी"
]

hinglish_bad = [
    "chutiya", "kamina"
]

all_bad_words = list(set(
    english_bad +
    hindi_bad +
    hinglish_bad +
    csv_bad_words
))

# ==========================================
# FUNCTIONS
# ==========================================
def is_hinglish(text):
    text = text.lower()
    return any(word in text for word in hinglish_keywords)


def contains_hindi(text):
    return any("\u0900" <= c <= "\u097F" for c in text)


def translate_text(text, target):
    try:
        return translator.translate(
            text,
            dest=target
        ).text
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
    st.success(
        f"CSV Loaded Successfully ({len(csv_bad_words)} words)"
    )
else:
    st.warning(
        "CSV not found. Keep english_hindi_badwords_500.csv in same folder."
    )

# Text input
text = st.text_area(
    "Enter Text Here:",
    height=220,
    placeholder="Type your paragraph here..."
)

# Language
output_lang = st.selectbox(
    "Choose Output Summary Language",
    ["English", "Hindi", "Hinglish"]
)

# ==========================================
# BUTTON
# ==========================================
if st.button("Generate Summary 🚀"):

    if text.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    original = text
    working = text

    # Hinglish → English
    if is_hinglish(working):
        working = translate_text(
            working,
            "en"
        )

    # Hindi → English
    elif contains_hindi(working):
        working = translate_text(
            working,
            "en"
        )

    # Summarize
    with st.spinner("Generating summary..."):
        summary = summarize_text(working)

    # Output Language
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
    bad_words = detect_bad_words(original)

    st.subheader("🛡 Language Safety Check")

    if bad_words:
        st.error("⚠ Unparliamentary Language Detected")
        st.write(bad_words)
    else:
        st.info("No unparliamentary language detected.")

# Footer
st.markdown("---")
st.caption("Developed using Streamlit + NLP + Transformers")
