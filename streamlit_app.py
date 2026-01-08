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
# CSS — MODERN UI (Force Light Theme)
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Force Light Theme Base */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

html, body, [class*="css"], .stMarkdown, .stText, p, div, label, span, h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #0f172a !important;
}

/* Fix Inputs (Text Area, Inputs) which might default to dark mode styles */
div[data-baseweb="textarea"], div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

textarea, input {
    color: #0f172a !important;
    background-color: #ffffff !important;
    caret-color: #2563eb !important;
}

textarea::placeholder, input::placeholder {
    color: #94a3b8 !important;
}

/* Radio Buttons & Checkboxes */
div[role="radiogroup"] label {
    color: #0f172a !important;
}

/* Headings specific overrides */
h1 {
    font-weight: 800 !important;
    background: -webkit-linear-gradient(135deg, #2563eb, #1e40af);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 12px -1px rgba(37, 99, 235, 0.3) !important;
}
div.stButton > button:active {
    transform: translateY(0);
}

/* Expander & Status */
div[data-testid="stStatusWidget"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}
div[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Code Editorish Look for the HTML Editor */
.stTextArea textarea {
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace !important;
    font-size: 14px !important;
}

/* Live Preview Box */
.preview-box {
    background: #ffffff;
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); /* Soft shadow */
    font-family: 'Segoe UI', serif; 
    line-height: 1.8;
    color: #1e293b !important;
}

/* Links in Preview */
.preview-box a {
    color: #2563eb !important;
    text-decoration: underline;
    font-weight: 500;
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
    Acts as a smart filter.
    """
    model = genai.GenerativeModel(LLM_MODEL)
    prompt = f"""
    Analyze the following text and identify all 'Company' or 'Organization' names mentioned.
    Include nicknames or variations (e.g., 'Canyon Energy' for 'Canyon Magnet Energy').
    Return ONLY a JSON list of strings.
    
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
    except Exception:
        # st.error(f"NER Error: {e}")
        return [], {"prompt_tokens":0, "completion_tokens":0}

def match_entities_to_db(entities, meta):
    """
    Step 2: Map extracted names to our existing URLs in 'meta'.
    """
    forced = {}
    
    # Pre-index meta by normalized title for O(1) lookup
    meta_index = {}
    word_index = {}
    
    for row in meta:
        title = entity_title_from_url(row["url"])
        norm_title = normalize(title)
        meta_index[norm_title] = row
        
        # Index unique significant words for partial matching
        for w in norm_title.split():
            if len(w) >= 4:
                if w not in word_index: word_index[w] = []
                word_index[w].append(row)

    for ent in entities:
        if not isinstance(ent, str): continue
        n_ent = normalize(ent)
        
        # 1. Exact match
        if n_ent in meta_index:
            row = meta_index[n_ent]
            forced[row["url"]] = row
            continue
            
        # 2. Partial / Keyword Match
        # If entity is "Canyon Energy", we look for "canyon" (len=6) and "energy" (len=6)
        # "Canyon" might point to "Canyon Magnet Energy".
        
        candidates = []
        for w in n_ent.split():
            if len(w) >= 4 and w in word_index:
                candidates.extend(word_index[w])
        
        # If we found candidates, check if they are "good" matches (overlapping words)
        if candidates:
            # unique candidates
            cand_map = {c['url']: c for c in candidates}
            for url, row in cand_map.items():
                row_title_norm = normalize(entity_title_from_url(url))
                
                # Jaccardish containment: do they share significant words?
                s1 = set(w for w in n_ent.split() if len(w) > 3)
                s2 = set(w for w in row_title_norm.split() if len(w) > 3)
                
                if s1 & s2: # If they share at least one significant word
                    forced[url] = row

    return forced

def build_candidates_v2(draft: str, mat, meta, forced_map):
    sentences = split_sentences(draft)
    items = []
    if not sentences: return []

    try:
        sent_embeddings = batch_embed(sentences)
    except Exception:
        return []

    all_scores = (mat @ sent_embeddings.T).T 
    
    for i, sent in enumerate(sentences):
        pid = i + 1
        scores = all_scores[i]
        sent_norm = normalize(sent)
        
        this_sent_candidates = {}
        
        # 1. Add Forced matches IF they (or their distinct parts) allow it
        # We rely on the LLM later to decide if it's REALLY a match, 
        # so we can be generous here.
        for url, row in forced_map.items():
            # Only add if the entity (or part of it) is plausibly in the sentence
            # Checking full title might fail for "Canyon Energy" vs "Canyon Magnet"
            # So checking the KEY is better.
            
            # Simple heuristic: If any significant word of the entity title is in the sentence
            title = entity_title_from_url(url)
            t_words = [w for w in normalize(title).split() if len(w) > 3]
            if any(w in sent_norm for w in t_words):
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
            if len(this_sent_candidates) >= 20: # Increased limit
                break
        
        items.append({
            "phrase_id": pid,
            "sentence": sent,
            "candidates": list(this_sent_candidates.values())
        })

    return items

def call_llm_autolink(draft: str, phrases: list):
    model = genai.GenerativeModel(LLM_MODEL)
    
    payload = {
        "draft": draft,
        "phrases": phrases,
        "rules": {
            "max_links": MAX_LINKS_TOTAL,
            "strategy": "Select the correct URL for entities in the text. Return a list of insertions.",
            "output_format": [{"anchor": "exact text to link", "url": "url"}]
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
        if isinstance(raw, list): return {"insertions": raw}, usage
        return raw, usage
    except Exception:
        return {"insertions": []}, {"prompt_tokens":0, "completion_tokens":0}

def apply_insertions(html: str, insertions: list) -> str:
    """
    Robust insertion that prevents replacing text inside existing HTML tags.
    """
    # 1. De-duplicate and validate
    valid_ins = []
    seen = set()
    for ins in insertions:
        u, a = ins.get('url'), ins.get('anchor')
        if not u or not a: continue
        if u in seen: continue
        seen.add(u)
        valid_ins.append(ins)
    
    # Sort by anchor length (longest first)
    valid_ins.sort(key=lambda x: len(x['anchor']), reverse=True)
    
    # 2. Split into tokens (tags vs text)
    # This splits by tags, e.g. "Foo <b>Bar</b>" -> ['Foo ', '<b>', 'Bar', '</b>', '']
    tokens = re.split(r'(<[^>]+>)', html)
    
    # 3. Perform replacement ONLY on text tokens
    for ins in valid_ins:
        anchor = ins['anchor']
        url = ins['url']
        pattern = re.compile(re.escape(anchor), re.IGNORECASE)
        
        found = False
        for i, token in enumerate(tokens):
            # If it's a tag (starts with <), skip
            if token.startswith('<'):
                continue
                
            # If we already found this link (prevent duplicates), skip? 
            # (Requirement: "no_duplicate_urls" usually implies 1 link per URL globally)
            # If we want to link ALL occurrences, remove `found` check.
            # But normally we link the first occurrence.
            if found: break
            
            # Search in text token
            if pattern.search(token):
                # Replace ONLY the first occurrence in this token
                # Check if it's already linked? No, `token` is pure text outside tags.
                # BUT, wait - if we have "Safe <a href='...'>Safe</a>", tokens are "Safe ", "<a>", "Safe", "</a>".
                # We won't accidentally link inside the href.
                
                # We use a placeholder to avoid re-matching inside this loop iteration accidentally
                # though regex logic on static string is fine, but placeholder is safer for final reassembly
                
                new_token = pattern.sub(f'__LINK_PLACEHOLDER_{url}__', token, count=1)
                if new_token != token:
                    tokens[i] = new_token
                    found = True # Link matched for this URL
                    
    # 4. Reassemble and fill placeholders
    # This avoids nested tag issues completely because we never insert valid HTML until the very end.
    final_html = "".join(tokens)
    
    for ins in valid_ins:
        url = ins['url']
        anchor = ins['anchor']
        placeholder = f'__LINK_PLACEHOLDER_{url}__'
        link_html = f'<a href="{url}">{anchor}</a>'
        final_html = final_html.replace(placeholder, link_html)
        
    return final_html

def remove_link_from_html(html, url_to_remove):
    # Simplistic removal: replace <a href="TARGET">Anchor</a> with Anchor
    # Regex needs to be robust to attributes order, but we control the insertion format mostly.
    # We'll valid lenient regex.
    pattern = re.compile(rf'<a\s+href="{re.escape(url_to_remove)}"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    return pattern.sub(r'\1', html)

def extract_links_for_ui(html):
    return [
        {"url": m.group(1), "anchor": m.group(2)}
        for m in re.finditer(r'<a href="([^"]+)">(.*?)</a>', html)
    ]

# ==================================================
# UI Layout
# ==================================================

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
        cost_out = (st.session_state.total_output_tokens / 1_000_000) * 2.50 
        st.session_state.total_cost = cost_in + cost_out

        st.session_state.result = {
            "html": final_html,
            "phrases": phrases
        }
        
        status.update(label="✅ Done!", state="complete", expanded=False)

def delete_link_callback(url_to_remove):
    if st.session_state.result:
        current_html = st.session_state.result["html"]
        new_html = remove_link_from_html(current_html, url_to_remove)
        st.session_state.result["html"] = new_html
        # We can safely update editor_content here because callbacks run before the script re-runs
        st.session_state.editor_content = new_html

def render_output_section():
    st.divider()
    st.subheader("🎉 Result")
    
    if "editor_content" not in st.session_state:
        st.session_state.editor_content = st.session_state.result["html"]
    
    # 3 Column Layout for Editor, Preview, Links
    c1, c2, c3 = st.columns([0.35, 0.35, 0.3])
    
    with c1:
        st.markdown("#### ✏️ Editor (HTML)")
        st.text_area(
            "Edit Code", 
            value=st.session_state.result["html"],
            key="editor_content",
            height=500,
            on_change=update_result_html
        )
        
    with c2:
        st.markdown("#### 👁️ Live Preview")
        html_content = st.session_state.result["html"]
        st.markdown(
            f'<div class="preview-box">{html_content}</div>', 
            unsafe_allow_html=True
        )

    with c3:
        st.markdown("#### 🔗 Manage Links")
        links = extract_links_for_ui(st.session_state.result["html"])
        if not links:
            st.info("No links found.")
        else:
            for i, l in enumerate(links):
                c_del, c_txt = st.columns([0.2, 0.8])
                with c_del:
                    st.button(
                        "🗑️", 
                        key=f"del_{i}", 
                        help=f"Remove link to {l['url']}",
                        on_click=delete_link_callback,
                        args=(l['url'],)
                    )
                with c_txt:
                    st.markdown(f"[{l['anchor']}]({l['url']})")

if __name__ == "__main__":
    main()