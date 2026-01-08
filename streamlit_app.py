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
    Also adds a Regex fallback to catch capitalized words (potential names) that LLM might miss.
    """
    # 1. LLM Extraction
    model = genai.GenerativeModel(LLM_MODEL)
    prompt = f"""
    Analyze the following text and identify all 'Company' or 'Organization' names mentioned.
    Include nicknames or variations (e.g., 'Canyon Energy' for 'Canyon Magnet Energy').
    Return ONLY a JSON list of strings.
    
    Text:
    {draft[:8000]}
    """
    
    entities = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    try:
        resp = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        entities = json.loads(resp.text)
        usage = {
            "prompt_tokens": resp.usage_metadata.prompt_token_count,
            "completion_tokens": resp.usage_metadata.candidates_token_count
        }
    except Exception:
        pass

    # 2. Regex Fallback (Heuristic)
    # Find words starting with Capital letter, containing letters/numbers.
    # Exclude common stopwords.
    regex_matches = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', draft)
    
    # Simple stoplist for the regex (since we don't want to fuzzy match 'The' against 'The Not Company')
    STOPWORDS = {
        "The", "A", "And", "Or", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "Up", "Out", 
        "It", "Is", "Are", "Was", "Were", "Be", "Been", "Has", "Have", "Had", "Do", "Does", "Did",
        "But", "So", "If", "While", "When", "Where", "Why", "How", "All", "Any", "Some", "No", "Not", 
        "Yes", "We", "You", "They", "He", "She", "It", "My", "Your", "Our", "Their", "His", "Her",
        "This", "That", "These", "Those", "Just", "More", "Most", "Other", "Such", "New", "Good", "High",
        "Our", "Your", "My", "Their", "His", "Her", "Its"
    } # "Not" is in stopwords... wait. "NotCo" is CamelCase. "Not" is just "Not".
    # User wants "NotCo". 
    # "Not" is in STOPWORDS. "NotCo" is NOT in STOPWORDS.
    
    # Filter and add
    for m in regex_matches:
        if len(m) >= 3 and m not in STOPWORDS and m not in entities:
             entities.append(m)
             
    # Also catch CamelCase explicitly? "NotCo" is matched by [A-Z][a-zA-Z0-9]+
    
    return entities, usage

import difflib

def match_entities_to_db(entities, meta):
    """
    Step 2: Map extracted names to our existing URLs in 'meta'.
    Now with fuzzy matching.
    """
    forced = {}
    
    # Pre-index meta
    meta_index = {} # Exact slug
    word_index = {} # Word -> list of rows
    all_titles = [] # For fuzzy scanning
    
    for row in meta:
        title = entity_title_from_url(row["url"])
        norm_title = normalize(title)
        
        meta_index[norm_title] = row
        all_titles.append((norm_title, row))
        
        # Index unique significant words
        # Lowered threshold to 3 to catch "Not", "Bio", "Hax", "Fab"
        for w in norm_title.split():
            if len(w) >= 3:
                if w not in word_index: word_index[w] = []
                word_index[w].append(row)

    for ent in entities:
        if not isinstance(ent, str): continue
        n_ent = normalize(ent)
        if len(n_ent) < 2: continue
        
        # 1. Exact match
        if n_ent in meta_index:
            row = meta_index[n_ent]
            forced[row["url"]] = {"row": row, "alias": ent}
            continue
            
        # 2. Token Overlap (Set Intersection) with CamelCase handling
        # "NotCo" -> "Not Co"
        spaced_ent = re.sub(r'([a-z])([A-Z])', r'\1 \2', ent)
        n_ent_spaced = normalize(spaced_ent)
        
        candidates = []
        # Check both the original run-together words and spaced words
        # "notco" might be in index? Unlikely. "not" "co" likely are.
        ent_words = set([w for w in n_ent.split() if len(w) >= 3] + 
                        [w for w in n_ent_spaced.split() if len(w) >= 3])
        
        for w in ent_words:
            if w in word_index:
                candidates.extend(word_index[w])
        
        # 3. Fuzzy Check on Candidates + Global Fuzzy fallback
        best_score = 0
        best_row = None
        
        # Helper: is the first word of entity matching the first word of target?
        # ent: "Canyon Energy" -> "canyon"
        # target: "Canyon Magnet Energy" -> "canyon"
        # ent: "NotCo" -> "not co" -> "not"
        # target: "The Not Company" -> "the" (mismatch) -> we want to skip "the"
        
        ent_tokens = n_ent_spaced.split()
        ent_first = ent_tokens[0] if ent_tokens else ""
        
        # Dedup candidates
        unique_cands = {c['url']: c for c in candidates}
        
        candidates_to_check = list(unique_cands.items()) if unique_cands else []
        
        for url, row in candidates_to_check:
            t_norm = normalize(entity_title_from_url(url))
            
            # Base SCORE: Fuzzy match
            score = difflib.SequenceMatcher(None, n_ent_spaced, t_norm).ratio()
            
            # Smart First Word Match
            # Strip "the " from title for comparison
            t_clean = re.sub(r'^the\s+', '', t_norm)
            t_first = t_clean.split()[0] if t_clean else ""
            
            # BONUS 1: First Word appears in title
            if len(ent_first) >= 3 and ent_first in t_norm:
                score += 0.15 # Boost
                
            # BONUS 2: First Word is the SAME as Title's First Word (ignoring 'The')
            if len(ent_first) >= 3 and ent_first == t_first:
                score += 0.3 # Bigger Boost
            
            if score > best_score:
                best_score = score
                best_row = row
        
        # Threshold Logic
        threshold = 0.6
        # If we have a Strong First Word Match (bonus 2 applied), we trust it more
        # "Canyon" (ent) vs "Canyon Magnet" (title)
        # Score ~0.5 + 0.15 + 0.3 = 0.95
        if len(ent_first) >= 3: threshold = 0.45 
        
        if best_score > threshold: 
            forced[best_row["url"]] = {"row": best_row, "alias": ent}
            continue
            
        # 4. Fallback: If no candidates found via words (unlikely for "NotCo" if "Not" is indexed),
        # or if checking acronyms/variations. 
        # NotCo (5) vs The Not Company (15).
        # Substring check?
        
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
        
        # 1. Add Forced matches
        for url, data in forced_map.items():
            row = data['row']
            alias = data['alias'] # The name found by NER (e.g. "Canyon Energy")
            
            # Check if the alias (fuzzy) is in this sentence
            # We trust NER that 'alias' IS in the text somewhere.
            # But is it in *this* sentence?
            
            # Simple check: if a signficiant part of alias is in sent_norm
            alias_parts = [w for w in normalize(alias).split() if len(w) >= 3]
            is_in_sentence = False
            if not alias_parts:
                # Alias is short? "HAX"
                if normalize(alias) in sent_norm: is_in_sentence = True
            else:
                # If any significant word of alias matches
                if any(w in sent_norm for w in alias_parts):
                    is_in_sentence = True
            
            if is_in_sentence:
                 this_sent_candidates[url] = {
                    "url": url,
                    "title": entity_title_from_url(url),
                    "type": row.get("content_type", "company"),
                    "score": 1.0,
                    "is_strong_match": True,
                    "matched_via": alias 
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
            if len(this_sent_candidates) >= 20:
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
    Uses opaque placeholders to avoid recursive matching.
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
    
    # Sort by anchor length (longest first) to match specific phrases before general ones
    valid_ins.sort(key=lambda x: len(x['anchor']), reverse=True)
    
    # 2. Split into tokens (tags vs text)
    tokens = re.split(r'(<[^>]+>)', html)
    
    # Store replacements map: placeholder -> (url, anchor)
    replacements = {}
    
    # 3. Perform replacement ONLY on text tokens
    for i, ins in enumerate(valid_ins):
        anchor = ins['anchor']
        url = ins['url']
        # Use a short, opaque placeholder that won't trigger other matches
        pid = f"__L_{i}__" 
        replacements[pid] = f'<a href="{url}">{anchor}</a>'
        
        pattern = re.compile(re.escape(anchor), re.IGNORECASE)
        
        found_for_this_link = False
        
        for k, token in enumerate(tokens):
            # Skip tags or tokens that are already placeholders (though our PID is unique enough)
            if token.startswith('<') or token.startswith('__L_'):
                continue
                
            if found_for_this_link: break
            
            if pattern.search(token):
                # Replace FIRST occurrence in this token
                new_token = pattern.sub(pid, token, count=1)
                if new_token != token:
                    tokens[k] = new_token
                    found_for_this_link = True
                    
    # 4. Reassemble
    final_html = "".join(tokens)
    
    # 5. Swap placeholders for real links
    for pid, link_tag in replacements.items():
        final_html = final_html.replace(pid, link_tag)
        
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
        current_urls = {l['url'] for l in links}
        
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
                    
    # NEW: Potential Matches Section
    st.divider()
    with st.expander("💡 Potential Missed Links (High Confidence)", expanded=True):
        # Scan phrases for high score items NOT in current_urls
        phrases = st.session_state.result.get("phrases", [])
        missed = []
        seen_missed = set()
        
        for p in phrases:
            for c in p["candidates"]:
                if c['url'] in current_urls: continue
                if c['url'] in seen_missed: continue
                
                # Criteria for being "Potential":
                # 1. Strong Match (NER matched)
                # 2. High Semantic Score (e.g. > 0.70)
                if c.get("is_strong_match") or c.get("score", 0) > 0.72:
                    missed.append({
                        "url": c['url'],
                        "title": c['title'],
                        "score": c.get("score", 0),
                        "reason": c.get("matched_via", "Semantic Match"),
                        "sentence": p["sentence"]
                    })
                    seen_missed.add(c['url'])
        
        if not missed:
            st.write("No other high-confidence matches found.")
        else:
            # Sort by score
            missed.sort(key=lambda x: x['score'] if x['score'] < 2 else 1.0, reverse=True) # Sort semantics, put Strong (score 1.0) at top
            
            st.info("These pages matched your content but weren't automatically linked. You can add them manually in the editor.")
            
            for m in missed[:15]: # Show top 15
                score_display = "NER Match" if m.get("reason") != "Semantic Match" else f"Score: {m['score']:.2f}"
                st.markdown(f"**[{m['title']}]({m['url']})** — *{score_display}*")
                st.caption(f"Context: \"...{m['sentence'][:80]}...\"")
                st.markdown("---")


if __name__ == "__main__":
    main()