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
TOP_LINKS_PER_PHRASE = 3

# ==================================================
# Setup
# ==================================================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
st.set_page_config(page_title="Internal Linking Assistant", layout="wide")

# ==================================================
# Session state
# ==================================================
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "accepted" not in st.session_state:
    st.session_state.accepted = {}

# ==================================================
# CSS (stable, light)
# ==================================================
st.markdown("""
<style>
html, body, .stApp { background:#fff; color:#111; font-size:22px; }
.block-container { max-width:1500px; padding:3rem 0 4rem; }

div[data-baseweb="textarea"] textarea,
div[data-baseweb="input"] input {
    background:#fff !important;
    color:#111 !important;
    font-size:20px !important;
    line-height:1.75 !important;
    border-radius:14px !important;
    border:1px solid #d1d5db !important;
    padding:18px !important;
}

div.stButton > button {
    background:#2563eb !important;
    color:#fff !important;
    font-size:18px !important;
    font-weight:650 !important;
    border-radius:14px !important;
    padding:0.7rem 1.2rem !important;
    border:none !important;
}

div.stButton > button:hover { background:#1e40af !important; }

.box { border:1px solid #e5e7eb; border-radius:16px; padding:18px; background:#fafafa; }
.annotated { white-space:pre-wrap; font-size:20px; line-height:1.9; }
.smalltag { font-size:14px; padding:3px 10px; border-radius:999px; background:#f3f4f6; margin-right:10px; }
.phrase-title { font-size:20px; font-weight:750; margin-top:20px; }
.helper { font-size:17px; color:#6b7280; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Helpers
# ==================================================
@st.cache_resource
def load_embeddings():
    mat = np.load(EMB_PATH).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def embed(text):
    v = genai.embed_content(model=EMBED_MODEL, content=text)["embedding"]
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) >= 30]

def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())

def company_name_from_url(url):
    return normalize(url.rstrip("/").split("/")[-1].replace("-", " "))

def extract_anchor_exact(sentence, url):
    candidates = re.findall(r"[A-Za-z][^,.;:!?]*[.,;:!?]?", sentence)
    slug = company_name_from_url(url)

    for c in candidates:
        if any(w in slug for w in normalize(c).split()):
            return c.strip()

    return max(candidates, key=len).strip()

def apply_links_preserve_format(draft, accepted):
    out = draft
    for idx in sorted(accepted):
        data = accepted[idx]
        anchor = data["anchor"]
        url = data["url"]

        m = re.search(re.escape(anchor), out, flags=re.IGNORECASE)
        if not m:
            continue

        matched = out[m.start():m.end()]
        out = out[:m.start()] + f'<a href="{url}">{matched}</a>' + out[m.end():]

    return out

# ==================================================
# Load corpus
# ==================================================
mat, meta = load_embeddings()

# ==================================================
# UI
# ==================================================
st.markdown("## 🔗 Internal Linking Assistant")

draft = st.text_area("Draft", height=420, placeholder="Paste your draft here…")

if st.button("Find internal links"):
    sentences = split_sentences(draft)
    phrases = []

    for sent in sentences:
        q = embed(sent)
        scores = mat @ q
        sent_norm = normalize(sent)

        company_present = any(
            normalize(company_name_from_url(row["url"])) in sent_norm
            for row in meta if row["content_type"] == "company"
        )

        def type_priority(t: str) -> int:
            return {"company": 0, "event": 1, "post": 2}.get(t, 3)

        scored = []
        for i, row in enumerate(meta):
            score = float(scores[i])
            prio = type_priority(row.get("content_type", "")) if company_present else 0
            scored.append({"prio": prio, "score": score, "row": row})

        # ✅ key-based sort (never compares dicts to dicts)
        ranked = sorted(
            scored,
            key=lambda x: (x["prio"], -x["score"], x["row"].get("url", "")),
        )[:TOP_LINKS_PER_PHRASE]

        options = []
        for item in ranked:
            row = item["row"]
            anchor = extract_anchor_exact(sent, row["url"])
            options.append({"row": row, "anchor": anchor})

        phrases.append({"sentence": sent, "options": options})

    st.session_state.analysis = {"phrases": phrases}
    st.session_state.accepted = {}

# ==================================================
# Output
# ==================================================
if st.session_state.analysis:
    phrases = st.session_state.analysis["phrases"]

    st.markdown("### Review")
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        annotated = [f"({i}) {p['sentence']}" for i, p in enumerate(phrases, 1)]
        st.markdown("<div class='box annotated'>" + "\n".join(annotated) + "</div>", unsafe_allow_html=True)

    with col_b:
        final_html = apply_links_preserve_format(draft, st.session_state.accepted)
        st.text_area("Final HTML", final_html, height=360)

    st.markdown("### Suggested links")
    st.markdown('<div class="helper">⚠️ Anchor must exactly match the draft text (including punctuation)</div>', unsafe_allow_html=True)

    for i, p in enumerate(phrases, 1):
        st.markdown(f"<div class='phrase-title'>Phrase ({i})</div>", unsafe_allow_html=True)

        for j, opt in enumerate(p["options"]):
            row = opt["row"]
            anchor_key = f"anchor_{i}_{j}"
            accepted = i in st.session_state.accepted and st.session_state.accepted[i]["url"] == row["url"]

            c1, c2 = st.columns([1.2, 6])

            with c1:
                label = "Accepted" if accepted else "Accept"
                if st.button(label, key=f"btn_{i}_{j}"):
                    if accepted:
                        st.session_state.accepted.pop(i)
                    else:
                        anchor_val = st.session_state.get(anchor_key, opt["anchor"])
                        st.session_state.accepted[i] = {"url": row["url"], "anchor": anchor_val}
                    st.rerun()

            with c2:
                st.text_input("", value=opt["anchor"], key=anchor_key, label_visibility="collapsed")

            st.markdown(f"<span class='smalltag'>{row['content_type'].capitalize()}</span> {row['url']}", unsafe_allow_html=True)

        st.markdown("---")