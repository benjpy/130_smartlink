import csv
import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# Config
# ==================================================
BASE_DIR = "sosv_content"
EMB_PATH = f"{BASE_DIR}/embeddings.npy"
META_PATH = f"{BASE_DIR}/embeddings_meta.csv"
EMBED_MODEL = "models/embedding-001"

TOP_K = 12
SENT_BATCH_SIZE = 64

# ==================================================
# Setup
# ==================================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="Internal Linking Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# Session state
# ==================================================
if "accepted_links" not in st.session_state:
    st.session_state.accepted_links = {}

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

# ==================================================
# UI — BIG, EDITOR-FRIENDLY TYPOGRAPHY
# ==================================================
st.markdown(
    """
    <style>
    html, body, .stApp {
        background:#ffffff !important;
        color:#111827 !important;
        font-size:19px;
    }

    .block-container {
        max-width:1400px;
        padding:3rem 0 4rem 0;
    }

    h1,h2,h3 {
        font-weight:700;
        color:#111827;
    }

    textarea, input {
        font-size:18px !important;
        line-height:1.7 !important;
        background:#ffffff !important;
        color:#111827 !important;
        border:1px solid #e5e7eb !important;
        border-radius:14px !important;
        padding:16px !important;
    }

    button[kind="primary"] {
        background:#2563eb!important;
        color:white!important;
        border-radius:14px!important;
        padding:0.8rem 1.8rem!important;
        font-weight:600!important;
        font-size:17px!important;
    }

    div.stButton > button {
        background:#ffffff!important;
        color:#2563eb!important;
        border:1px solid #c7d2fe!important;
        border-radius:12px!important;
        font-weight:600!important;
        font-size:17px!important;
    }

    .card {
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:18px;
        padding:20px 22px;
        margin-bottom:18px;
        box-shadow:0 2px 8px rgba(0,0,0,.05);
        font-size:18px;
    }

    .badge {
        font-size:14px;
        padding:6px 14px;
        border-radius:999px;
        background:#f3f4f6;
        color:#374151;
        margin-right:10px;
    }

    .accepted-pill {
        font-size:14px;
        padding:6px 14px;
        border-radius:999px;
        background:#ecfdf5;
        color:#065f46;
        border:1px solid #a7f3d0;
        display:inline-block;
        margin-bottom:10px;
    }

    .helper {
        font-size:16px;
        color:#6b7280;
        margin-bottom:12px;
    }

    .annotated {
        white-space:pre-wrap;
        font-size:18px;
        line-height:1.8;
        border:1px dashed #d1d5db;
        border-radius:14px;
        padding:18px;
        background:#fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# Helpers — embeddings
# ==================================================
@st.cache_resource
def load_embeddings():
    mat = np.load(EMB_PATH).astype(np.float32)
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)

def embed_query(text):
    emb = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def split_sentences(text):
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [x.strip() for x in s if len(x.strip()) >= 30]

def embed_sentences(sentences):
    vecs = []
    for i in range(0, len(sentences), SENT_BATCH_SIZE):
        batch = sentences[i:i+SENT_BATCH_SIZE]
        emb = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="retrieval_query",
        )["embedding"]
        arr = np.array(emb, dtype=np.float32)
        vecs.append(normalize(arr))
    return np.vstack(vecs)

def best_sentence(target_vec, sent_vecs, sentences):
    scores = sent_vecs @ target_vec
    i = int(np.argmax(scores))
    return sentences[i], i

# ==================================================
# Anchor extraction
# ==================================================
STOPWORDS = {
    "we","our","us","you","your","they","them","their","i","me","my",
    "this","that","these","those","a","an","the","and","or","but",
    "to","of","in","on","for","with","as","at","by","from","into",
    "is","are","was","were","be","been","being","it","its",
    "looking","join","make","made","get","getting","find","help",
    "build","support","create"
}

def extract_candidate_phrases(sentence):
    words = re.findall(r"[A-Za-z][A-Za-z\\-]+", sentence)
    phrases, current = [], []

    for w in words:
        if w.lower() in STOPWORDS:
            if current:
                phrases.append(" ".join(current))
            current = []
            continue
        if len(w) >= 4:
            current.append(w)
        else:
            if current:
                phrases.append(" ".join(current))
            current = []

    if current:
        phrases.append(" ".join(current))

    phrases = list(set(phrases))
    phrases.sort(key=len, reverse=True)
    return phrases

def extract_anchor(sentence, target_url):
    phrases = extract_candidate_phrases(sentence)
    if not phrases:
        return sentence.split()[0]

    slug = target_url.rstrip("/").split("/")[-1].replace("-", " ").lower()

    for p in phrases:
        if p.lower() in slug or any(w in slug for w in p.lower().split()):
            return p

    for p in phrases:
        if len(p.split()) >= 2:
            return p

    return phrases[0]

# ==================================================
# HTML builder
# ==================================================
def build_html(draft, accepted):
    paragraphs = [p.strip() for p in draft.split("\n") if p.strip()]
    html = []
    applied = {k: False for k in accepted}

    for p in paragraphs:
        updated = p
        for url, info in accepted.items():
            anchor = info["anchor"]
            m = re.search(re.escape(anchor), updated, flags=re.IGNORECASE)
            if not m:
                continue

            matched = updated[m.start():m.end()]
            link = f'<a href="{url}">{matched}</a>'
            updated = updated[:m.start()] + link + updated[m.end():]
            applied[url] = True

        html.append(f"<p>{updated}</p>")

    not_applied = [(u, accepted[u]["anchor"]) for u, ok in applied.items() if not ok]
    return "\n".join(html), not_applied

# ==================================================
# Load corpus
# ==================================================
mat, meta = load_embeddings()
matn = normalize(mat)

# ==================================================
# UI
# ==================================================
st.markdown("## 🔗 Internal Linking Assistant")

left, right = st.columns(2, gap="large")

with left:
    st.markdown("### Draft")
    draft = st.text_area("", height=420, placeholder="Paste or edit your draft…")

with right:
    st.markdown("### Final HTML")
    if st.session_state.accepted_links:
        html_preview, not_applied = build_html(draft, st.session_state.accepted_links)
    else:
        html_preview, not_applied = "<!-- Accept links to generate HTML -->", []

    st.text_area("", html_preview, height=420)

    if not_applied:
        st.warning("Some accepted anchors were not found verbatim in the draft.")

if st.button("Find internal link opportunities", type="primary"):
    st.session_state.analysis_done = True
    st.session_state.analysis_cache = {}
    st.session_state.accepted_links = {}

# ==================================================
# Analysis
# ==================================================
if st.session_state.analysis_done and draft.strip():

    if not st.session_state.analysis_cache:
        sentences = split_sentences(draft)
        sent_vecs = embed_sentences(sentences)

        q = embed_query(draft)
        scores = matn @ q
        top_idx = np.argsort(-scores)[:TOP_K]

        suggestions = []
        for i in top_idx:
            sent, sent_idx = best_sentence(matn[i], sent_vecs, sentences)
            suggestions.append({
                "meta": meta[i],
                "sentence": sent,
                "sentence_idx": sent_idx,
            })

        suggestions.sort(key=lambda x: x["sentence_idx"])

        st.session_state.analysis_cache = {
            "sentences": sentences,
            "suggestions": suggestions,
        }

    suggestions = st.session_state.analysis_cache["suggestions"]
    sentences = st.session_state.analysis_cache["sentences"]

    st.markdown("## Suggested links")
    st.markdown(
        '<div class="helper">⚠️ Anchor must exactly match the draft text (including punctuation)</div>',
        unsafe_allow_html=True,
    )

    emojis = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩","⑪","⑫"]

    # -------- Annotated draft (review only) --------
    annotated = []
    for idx, s in enumerate(sentences):
        marker = ""
        for j, sug in enumerate(suggestions):
            if sug["sentence"] == s:
                marker = f"{emojis[j]} "
        annotated.append(marker + s + "\n")

    st.markdown("### Annotated draft (review only)")
    st.markdown(
        f'<div class="annotated">{"\n\n".join(annotated)}</div>',
        unsafe_allow_html=True,
    )

    # -------- Suggested links list --------
    for idx, item in enumerate(suggestions):
        row = item["meta"]
        url = row["url"]
        sent = item["sentence"]
        anchor_suggested = extract_anchor(sent, url)

        col_btn, col_card = st.columns([1.3, 5])

        with col_btn:
            st.markdown(f"### {emojis[idx]}")
            if url in st.session_state.accepted_links:
                st.markdown('<div class="accepted-pill">✓ Accepted</div>', unsafe_allow_html=True)
                if st.button("Undo", key=f"undo_{idx}"):
                    st.session_state.accepted_links.pop(url, None)
                    st.rerun()
            else:
                anchor_edit = st.text_input(
                    "Anchor",
                    value=anchor_suggested,
                    key=f"anchor_edit_{idx}",
                    label_visibility="collapsed",
                )
                if st.button("Accept", key=f"accept_{idx}"):
                    st.session_state.accepted_links[url] = {
                        "anchor": anchor_edit.strip(),
                        "sentence": sent,
                        "content_type": row["content_type"],
                    }
                    st.rerun()

        with col_card:
            preview = sent.replace(anchor_suggested, f"<strong>{anchor_suggested}</strong>", 1)
            st.markdown(
                f"""
                <div class="card">
                    <span class="badge">{row["content_type"].capitalize()}</span>
                    <strong>{url}</strong>
                    <div style="margin-top:14px;">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )