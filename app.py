import streamlit as st
from transformers import pipeline
from googletrans import Translator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import pandas as pd
import re

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Multilingual Summarization Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    translator = Translator()
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    return translator, summarizer

translator, summarizer = load_models()

# ==========================================
# LOAD CSV BAD WORDS
# Keep file in same folder:
# english_hindi_badwords_500.csv
# ==========================================
csv_file = "english_hindi_badwords_500.csv"

@st.cache_data
def load_bad_words():
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")

        # Only use 'word' column
        words = (
            df["word"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        return list(set(words))

    except Exception:
        return []

csv_bad_words = load_bad_words()

# ==========================================
# HINGLISH KEYWORDS
# ==========================================
hinglish_keywords = [
    "kal", "hai", "mera", "tum",
    "hum", "khelenge", "ayega",
    "aayega", "bhai", "acha"
]

# ==========================================
# BACKUP BAD WORDS
# ==========================================
english_bad = ["idiot", "stupid", "damn"]
hindi_bad = ["बेवकूफ", "हरामी"]
hinglish_bad = ["chutiya", "kamina"]

# Final combined words
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

def hinglish_to_hindi(text):
    try:
        return transliterate(
            text,
            sanscript.ITRANS,
            sanscript.DEVANAGARI
        )
    except:
        return text

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

def detect_unparliamentary(text):
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
st.title("🤖 Multilingual Summarization Chatbot")
st.write(
    "Supports **English, Hindi, Hinglish** + "
    "Bad Word Detection using CSV Dataset"
)

if csv_bad_words:
    st.success(
        f"CSV Loaded Successfully "
        f"({len(csv_bad_words)} words)"
    )
else:
    st.warning(
        "CSV not found. "
        "Place english_hindi_badwords_500.csv "
        "in same folder."
    )

text = st.text_area(
    "Enter your text here:",
    height=220
)

output_lang = st.selectbox(
    "Select Summary Language",
    ["English", "Hindi", "Hinglish"]
)

# ==========================================
# BUTTON
# ==========================================
if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("Please enter text first.")
        st.stop()

    original = text
    working = text

    # Hinglish -> English
    if is_hinglish(working):
        working = translate_text(
            working,
            "en"
        )

    # Hindi -> English
    elif contains_hindi(working):
        working = translate_text(
            working,
            "en"
        )

    # Summarize in English
    summary = summarize_text(working)

    # Convert output language
    if output_lang == "Hindi":
        summary = translate_text(
            summary,
            "hi"
        )

    elif output_lang == "Hinglish":
        summary = translate_text(
            summary,
            "hi"
        )

        summary = (
            summary
            .replace("है", "hai")
            .replace("और", "aur")
            .replace("का", "ka")
        )

    # Show summary
    st.subheader("Summary")
    st.success(summary)

    # Detect bad words
    bad_words = detect_unparliamentary(original)

    st.subheader("Language Safety Check")

    if bad_words:
        st.error(
            "⚠ Unparliamentary Language Detected!"
        )
        st.write(bad_words)
    else:
        st.info(
            "No unparliamentary language detected."
        )