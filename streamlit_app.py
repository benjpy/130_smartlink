import csv
import os
import re
import json
import time
import numpy as np
import streamlit as st
import warnings

warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import google.generativeai as genai

# ==================================================
# Config
# ==================================================
BASE_DIR = "sosv_content"
EMB_PATH = f"{BASE_DIR}/embeddings.npy"
META_PATH = f"{BASE_DIR}/embeddings_meta.csv"

EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-flash"

TOP_CANDIDATES_PER_SENTENCE = 10
MAX_SENTENCES = 40
MAX_LINKS_TOTAL = 25

# ==================================================
# Setup
# ==================================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing Gemini API Key. Please set it in .env or Streamlit Secrets.")
    st.stop()
genai.configure(api_key=api_key)

st.set_page_config(
    page_title="SmartLink — AI Auto-Linker",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Session state
# ==================================================
if "result" not in st.session_state:
    st.session_state.result = None
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ==================================================
# CSS — MODERN UI
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

/* Headings */
h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
    letter-spacing: -0.025em;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    transition: all 0.2s;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
}

/* Text Areas */
div[data-baseweb="textarea"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 2px;
}
div[data-baseweb="textarea"]:focus-within {
    border-color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
}

/* Status Container */
div[data-testid="stStatusWidget"] {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #f1f5f9;
}

/* Cards/Containers */
.css-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

/* Custom classes for output */
.preview-box {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    font-family: serif; /* Simulate reader view */
    line-height: 1.8;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# Helpers: Data & Embedding
# ==================================================
@st.cache_resource
def load_data():
    if not os.path.exists(EMB_PATH) or not os.path.exists(META_PATH):
        return None, None
        
    mat = np.load(EMB_PATH).astype(np.float32)
    mat = np.nan_to_num(mat, copy=False)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.divide(mat, norm, out=np.zeros_like(mat), where=norm!=0)
    
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def batch_embed(texts: list) -> np.ndarray:
    if not texts: return np.array([])
    resp = genai.embed_content(model=EMBED_MODEL, content=texts)
    embeddings = resp["embedding"]
    mat = np.array(embeddings, dtype=np.float32)
    mat = np.nan_to_num(mat, copy=False)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    mat = mat / norm
    return mat

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())

def split_sentences(text: str):
    # Quick heuristic split
    return [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) >= 20
    ][:MAX_SENTENCES]

def entity_title_from_url(url: str):
    return url.rstrip("/").split("/")[-1].replace("-", " ")

# ==================================================
# Core Logic: NER & Linking
# ==================================================

def identify_entities_with_llm(draft: str):
    """
    Step 1: Ask LLM to extract potential company names from the text.
    This acts as a smart filter before we even look at our database.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    prompt = f"""
    Analyze the following text and identify all 'Company' or 'Organization' names mentioned.
    Return ONLY a JSON list of strings. Do not include extra text.
    
    Text:
    {draft[:8000]}
    """
    
    try:
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        entities = json.loads(resp.text)
        
        usage = {
            "prompt_tokens": resp.usage_metadata.prompt_token_count,
            "completion_tokens": resp.usage_metadata.candidates_token_count
        }
        return entities, usage
    except Exception as e:
        # st.error(f"NER Error: {e}")
        return [], {"prompt_tokens":0, "completion_tokens":0}

def match_entities_to_db(entities, meta):
    """
    Step 2: Map extracted names to our existing URLs in 'meta'.
    """
    forced = {}
    
    # Pre-index meta by normalized title for O(1) lookup
    meta_index = {}
    for row in meta:
        title = entity_title_from_url(row["url"])
        norm_title = normalize(title)
        meta_index[norm_title] = row
        # distinct words index for partial
        for w in norm_title.split():
            if len(w) > 4: # index significant words
                if w not in meta_index: meta_index[w] = []
                if isinstance(meta_index[w], list): meta_index[w].append(row)

    for ent in entities:
        if not isinstance(ent, str): continue
        n_ent = normalize(ent)
        
        # 1. Exact match on full slug
        if n_ent in meta_index and isinstance(meta_index[n_ent], dict):
            row = meta_index[n_ent]
            forced[row["url"]] = row
            continue
            
        # 2. Heuristic partial match
        # If the entity is "Orbit Fab" and we have "orbit-fab", normalize matches.
        # If entity is "HAX" and we have "hax", matches.
        # Check against list buckets
        found = False
        parts = n_ent.split()
        for p in parts:
            if len(p) > 4 and p in meta_index and isinstance(meta_index[p], list):
                # We have candidates containing this word
                for candidate_row in meta_index[p]:
                    cand_title = entity_title_from_url(candidate_row["url"])
                    # Simple inclusion check
                    if normalize(cand_title) in n_ent or n_ent in normalize(cand_title):
                        forced[candidate_row["url"]] = candidate_row
                        found = True
                        break
            if found: break
            
    return forced

def build_candidates_v2(draft: str, mat, meta, forced_map):
    sentences = split_sentences(draft)
    items = []
    if not sentences: return []

    # Embed sentences
    try:
        sent_embeddings = batch_embed(sentences)
    except Exception as e:
        return []

    # Scores
    all_scores = (mat @ sent_embeddings.T).T 
    
    for i, sent in enumerate(sentences):
        pid = i + 1
        scores = all_scores[i]
        
        # 1. Add Forced/NER matches if present in this sentence
        sent_norm = normalize(sent)
        this_sent_candidates = {}
        
        for url, row in forced_map.items():
            title = entity_title_from_url(url)
            # Check if this forced entity is actually in this sentence string
            if normalize(title) in sent_norm:
                 this_sent_candidates[url] = {
                    "url": url,
                    "title": title,
                    "type": row.get("content_type", "company"),
                    "score": 1.0,
                    "is_strong_match": True
                }

        # 2. Add Top Semantic Matches
        top_idx = np.argsort(-scores)[:50]
        for idx in top_idx:
            row = meta[idx]
            url = row["url"]
            if url not in this_sent_candidates:
                this_sent_candidates[url] = {
                    "url": url,
                    "title": entity_title_from_url(url),
                    "type": row.get("content_type", "post"),
                    "score": float(scores[idx]),
                    "is_strong_match": False
                }
            if len(this_sent_candidates) >= 15:
                break
        
        items.append({
            "phrase_id": pid,
            "sentence": sent,
            "candidates": list(this_sent_candidates.values())
        })

    return items

def call_llm_autolink(draft: str, phrases: list):
    """
    Final decision maker.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    
    payload = {
        "draft": draft,
        "phrases": phrases,
        "rules": {
            "max_links": MAX_LINKS_TOTAL,
            "strategy": "Identify the best matching entities from the candidates list for the text. Prefer Companies. Link phrases that clearly refer to the candidate. ONLY return the JSON."
        }
    }

    try:
        resp = model.generate_content(
            json.dumps(payload),
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        raw = json.loads(resp.text)
        usage = {
            "prompt_tokens": resp.usage_metadata.prompt_token_count,
            "completion_tokens": resp.usage_metadata.candidates_token_count
        }
        
        if isinstance(raw, list):
            return {"insertions": raw}, usage
        return raw, usage
        
    except Exception as e:
        return {"insertions": []}, {"prompt_tokens":0, "completion_tokens":0}

def apply_insertions(html: str, insertions: list) -> str:
    # De-duplicate by URL
    seen_urls = set()
    cleaned_ins = []
    
    # Sort insertions by anchor length descending to prefer longer matches first
    # This prevents replacing "Foobar" inside "Foobar Inc" incorrectly if "Foobar" is processed first
    sorted_insertions = sorted(insertions, key=lambda x: len(x.get("anchor", "")), reverse=True)
    
    for ins in sorted_insertions:
        if ins['url'] in seen_urls: continue
        seen_urls.add(ins['url'])
        cleaned_ins.append(ins)
        
    for ins in cleaned_ins:
        anchor = ins.get("anchor", "").strip()
        url = ins.get("url", "").strip()
        if not anchor or not url: continue
        
        # Regex replacement to ensure we don't link inside existing tags
        # (Very basic protection)
        pattern = re.compile(re.escape(anchor), re.IGNORECASE)
        
        def repl(m):
            # Check if we are inside a tag (simplistic check: count < vs > before match)
            # This is not perfect but better than nothing
            # For this quick tool, we just do direct replacement 
            # assuming the draft is plain text initially
            return f'<a href="{url}">{m.group(0)}</a>'
            
        html = pattern.sub(repl, html, count=1)
        
    return html

# ==================================================
# UI Layout
# ==================================================

# Function to update session state when editor changes
def update_result_html():
    st.session_state.result["html"] = st.session_state.editor_content

def main():
    st.title("🔗 SmartLink")
    st.markdown("Automated internal linking powered by **Gemini 2.5** and semantic search.")
    
    # Layout
    col_input, col_controls = st.columns([0.65, 0.35])
    
    with col_input:
        st.subheader("📝 Draft Content")
        draft = st.text_area("Paste your article here", height=300, key="draft_input", label_visibility="collapsed", placeholder="New startup taking over the world...")

    with col_controls:
        st.subheader("⚙️ Controls")
        
        # Load Data
        mat, meta = load_data()
        if mat is None:
            st.error("⚠️ Knowledge base not found.")
            # st.stop() 
        else:
            st.success(f"📚 Knowledge Base: {len(meta)} ent.")
        
        links_mode = st.radio("Link Types", ["Mixed", "Companies Only", "Content Only"], horizontal=True)
        
        if st.button("✨ Auto-Link Draft", use_container_width=True):
            if not draft:
                st.toast("Please enter some text first!", icon="⚠️")
            else:
                run_autolink_process(draft, mat, meta)

    # Output Section
    if st.session_state.result:
        render_output_section()

    # Sidebar Footer
    with st.sidebar:
        st.markdown("### 📊 Session Stats")
        st.info(f"Cost: **${st.session_state.total_cost:.4f}**")
        st.text(f"In: {st.session_state.total_input_tokens:,}")
        st.text(f"Out: {st.session_state.total_output_tokens:,}")
        
        if st.button("Reset Stats"):
            st.session_state.total_cost = 0.0
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()

def run_autolink_process(draft, mat, meta):
    with st.status("🚀 Processing...", expanded=True) as status:
        
        # 1. NER
        st.write("🔍 Identifying Key Entities (LLM)...")
        entities, usage_ner = identify_entities_with_llm(draft)
        st.write(f"   — Found: {', '.join(entities)}")
        
        # 2. Match to DB
        st.write("📂 Mapping to Knowledge Base...")
        forced_map = match_entities_to_db(entities, meta)
        
        # 3. Candidates
        st.write("🧠 Building Semantic Candidates...")
        phrases = build_candidates_v2(draft, mat, meta, forced_map)
        
        # 4. Final Linking
        st.write("🔗 Generating Links...")
        resp, usage_link = call_llm_autolink(draft, phrases)
        
        # 5. Apply
        final_html = apply_insertions(draft, resp.get("insertions", []))
        
        # Cost Logic
        st.session_state.total_input_tokens += (usage_ner["prompt_tokens"] + usage_link["prompt_tokens"])
        st.session_state.total_output_tokens += (usage_ner["completion_tokens"] + usage_link["completion_tokens"])
        
        cost_in = (st.session_state.total_input_tokens / 1_000_000) * 0.30
        cost_out = (st.session_state.total_output_tokens / 1_000_000) * 2.50 # approx blended
        st.session_state.total_cost = cost_in + cost_out

        st.session_state.result = {
            "html": final_html,
            "phrases": phrases
        }
        
        status.update(label="✅ Donel!", state="complete", expanded=False)

def render_output_section():
    st.divider()
    st.subheader("🎉 Result")
    
    # Initialize editor content if strictly necessary, but `key` handles it mostly
    if "editor_content" not in st.session_state:
        st.session_state.editor_content = st.session_state.result["html"]
    
    c1, c2 = st.columns([0.5, 0.5])
    
    with c1:
        st.markdown("#### ✏️ Editor (HTML)")
        # We bind this textarea to `editor_content`
        # On change, `update_result_html` syncs it back to `result['html']`
        st.text_area(
            "Edit Code", 
            value=st.session_state.result["html"],
            key="editor_content",
            height=500,
            on_change=update_result_html
        )
        
    with c2:
        st.markdown("#### 👁️ Live Preview")
        # Use result['html'] which is kept in sync
        html_content = st.session_state.result["html"]
        st.markdown(
            f'<div class="preview-box">{html_content}</div>', 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()