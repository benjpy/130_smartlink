import csv
import os
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
# HARD FORCE LIGHT THEME (this is the key fix)
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* --- FORCE LIGHT MODE --- */
    :root {
        --background-color: #ffffff !important;
        --secondary-background-color: #ffffff !important;
        --text-color: #111827 !important;
    }

    .stApp {
        background: #ffffff !important;
        color: #111827 !important;
    }

    html, body {
        background: #ffffff !important;
        color: #111827 !important;
    }

    /* --- LAYOUT --- */
    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* --- TYPOGRAPHY --- */
    h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    h2 {
        font-size: 1.4rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        color: #111827;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2.5rem;
        max-width: 720px;
    }

    /* --- TEXTAREA --- */
    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 14px !important;
    }

    /* --- BUTTON --- */
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

    /* --- RESULT CARDS --- */
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
        font-weight: 600;
        color: #111827;
        word-break: break-all;
    }

    .score {
        margin-top: 6px;
        font-size: 13px;
        color: #6b7280;
    }

    /* Remove Streamlit footer & menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    mat = np.load(EMB_PATH).astype(np.float32)
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-12)

def embed_query(text: str) -> np.ndarray:
    emb = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

# --------------------------------------------------
# Load data
# --------------------------------------------------
mat, meta = load_embeddings()
matn = normalize(mat)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("## 🔗 Internal Linking Assistant")
st.markdown(
    '<div class="subtitle">'
    'Paste a draft blog post and discover the most relevant internal links '
    'across SOSV blog posts, portfolio companies, and events.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("### Draft content")

draft = st.text_area(
    "",
    height=240,
    placeholder="Paste your blog draft here…",
)

run = st.button("Find internal link opportunities", type="primary")

# --------------------------------------------------
# Results
# --------------------------------------------------
if run:
    if not draft.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analyzing content…"):
            q = embed_query(draft)
            scores = matn @ q
            idx = np.argsort(-scores)[:TOP_K]

        st.markdown("## Suggested links")

        for rank, i in enumerate(idx, 1):
            row = meta[i]
            score = scores[i]

            st.markdown(
                f"""
                <div class="card">
                    <div>
                        <span class="badge">{row["content_type"].capitalize()}</span>
                        <span class="url">{rank}. {row["url"]}</span>
                    </div>
                    <div class="score">
                        Relevance score: {score:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )