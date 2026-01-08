import csv
import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = "sosv_content"
EMB_PATH = f"{BASE_DIR}/embeddings.npy"
META_PATH = f"{BASE_DIR}/embeddings_meta.csv"
EMBED_MODEL = "models/embedding-001"

TOP_K = 12
SENT_BATCH_SIZE = 64  # sentence embedding batch size

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="Internal Linking Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# HARD FORCE LIGHT THEME + Modern Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --background-color: #ffffff !important;
        --secondary-background-color: #ffffff !important;
        --text-color: #111827 !important;
    }

    .stApp, html, body {
        background: #ffffff !important;
        color: #111827 !important;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 700;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2.5rem;
        max-width: 720px;
    }

    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 14px !important;
    }

    button[kind="primary"] {
        background: #2563eb !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background: #1e40af !important;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    .badge {
        display: inline-block;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f3f4f6;
        color: #374151;
        margin-right: 10px;
    }

    .url {
        font-weight: 650;
        color: #111827;
        word-break: break-all;
    }

    .score {
        margin-top: 6px;
        font-size: 13px;
        color: #6b7280;
    }

    .anchorline {
        margin-top: 10px;
        font-size: 15px;
        color: #111827;
    }

    .smallhint {
        font-size: 13px;
        color: #6b7280;
        margin-top: 6px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Helpers: loading + normalization
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    mat = np.load(EMB_PATH).astype(np.float32)
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-12)

# --------------------------------------------------
# Helpers: draft sentence splitting + embedding
# --------------------------------------------------
def split_sentences(text: str) -> list[str]:
    # simple, robust sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # keep only reasonably-sized sentences
    return [s.strip() for s in sentences if len(s.strip()) >= 30]

def embed_query(text: str) -> np.ndarray:
    emb = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def embed_sentences(sentences: list[str]) -> np.ndarray:
    # Gemini can embed a list; we batch to avoid payload limits
    all_vecs = []
    for start in range(0, len(sentences), SENT_BATCH_SIZE):
        batch = sentences[start:start + SENT_BATCH_SIZE]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="retrieval_query",
        )["embedding"]
        vecs = np.array(result, dtype=np.float32)
        vecs = normalize(vecs)
        all_vecs.append(vecs)
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)

def best_sentence_for_target(target_vec: np.ndarray, sent_vecs: np.ndarray, sentences: list[str]):
    scores = sent_vecs @ target_vec
    idx = int(np.argmax(scores))
    return sentences[idx], float(scores[idx])

# --------------------------------------------------
# Anchor extraction (safe MVP heuristic)
# --------------------------------------------------
def extract_anchor(sentence: str, target_url: str) -> str:
    # Try to derive something meaningful from the URL slug
    slug = target_url.rstrip("/").split("/")[-1].replace("-", " ").strip()
    slug_words = [w for w in slug.split() if len(w) >= 3]

    sent_lower = sentence.lower()
    # If any slug word appears verbatim in sentence, pick the longest matching word
    matches = [w for w in slug_words if w.lower() in sent_lower]
    if matches:
        # prefer longest (more specific)
        matches.sort(key=len, reverse=True)
        return matches[0]

    # Otherwise: pick first "capitalized token sequence" (rough brand/entity heuristic)
    tokens = sentence.split()
    caps = [t.strip(".,;:()[]{}\"'") for t in tokens if t[:1].isupper() and len(t) > 2]
    if caps:
        return caps[0]

    # Fallback: first 5 words
    clean_tokens = [t.strip() for t in tokens if t.strip()]
    return " ".join(clean_tokens[:5]) if clean_tokens else "this"

def highlight_first(sentence: str, phrase: str) -> str:
    # Bold the first occurrence (case-sensitive first, then case-insensitive)
    if phrase in sentence:
        return sentence.replace(phrase, f"**{phrase}**", 1)

    # case-insensitive fallback
    m = re.search(re.escape(phrase), sentence, flags=re.IGNORECASE)
    if not m:
        return sentence

    start, end = m.span()
    return sentence[:start] + "**" + sentence[start:end] + "**" + sentence[end:]

def esc_html_keep_md_bold(s: str) -> str:
    # We are rendering markdown-ish bold inside HTML via st.markdown(unsafe_allow_html=True)
    # So: we avoid escaping **. We do basic escaping for < and > only.
    return s.replace("<", "&lt;").replace(">", "&gt;")

def md_bold_to_html(s: str) -> str:
    # Convert **bold** to <strong>bold</strong> for HTML rendering
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

# --------------------------------------------------
# Load corpus embeddings
# --------------------------------------------------
mat, meta = load_embeddings()
matn = normalize(mat)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("## 🔗 Internal Linking Assistant")
st.markdown(
    '<div class="subtitle">'
    'Paste a draft blog post and discover the most relevant internal links across '
    'SOSV blog posts, portfolio companies, and events — with suggested anchor sentences.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("### Draft content")
draft = st.text_area("", height=240, placeholder="Paste your blog draft here…")
run = st.button("Find internal link opportunities", type="primary")

# --------------------------------------------------
# Results
# --------------------------------------------------
if run:
    if not draft.strip():
        st.warning("Please paste some text first.")
        st.stop()

    sentences = split_sentences(draft)
    if len(sentences) < 1:
        st.warning("Couldn’t split your draft into sentences. Add a few sentences with punctuation and try again.")
        st.stop()

    with st.spinner("Embedding draft sentences and searching your corpus…"):
        # Search corpus (draft-level)
        q = embed_query(draft)
        scores = matn @ q
        top_idx = np.argsort(-scores)[:TOP_K]

        # Sentence embeddings (for anchor discovery)
        sent_vecs = embed_sentences(sentences)

    st.markdown("## Suggested links & anchors")

    for rank, i in enumerate(top_idx, 1):
        row = meta[i]
        target_vec = matn[i]
        target_score = float(scores[i])

        # Find best sentence
        best_sent, sent_score = best_sentence_for_target(target_vec, sent_vecs, sentences)

        # Extract anchor phrase
        anchor = extract_anchor(best_sent, row["url"])
        highlighted = highlight_first(best_sent, anchor)

        # Render card
        safe_sentence = md_bold_to_html(esc_html_keep_md_bold(highlighted))

        st.markdown(
            f"""
            <div class="card">
                <div>
                    <span class="badge">{row["content_type"].capitalize()}</span>
                    <span class="url">{rank}. {row["url"]}</span>
                </div>
                <div class="score">
                    Target relevance: {target_score:.3f} · Best sentence match: {sent_score:.3f}
                </div>
                <div class="anchorline">
                    {safe_sentence}
                </div>
                <div class="smallhint">
                    Suggested anchor: <strong>{anchor}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )