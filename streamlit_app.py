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
# CSS — PREMIUM UI (Modern, "Fancy")
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --secondary: #ec4899;
    --bg-color: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
}

/* Base App */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif;
    font-size: 20px !important; /* Huge Base Font */
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-main) !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-weight: 800 !important;
    font-size: 4rem !important; /* MASSIVE Title */
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 1rem !important;
    margin-bottom: 3rem !important;
}

h2 { font-size: 2.5rem !important; font-weight: 700 !important; margin-top: 2.5rem !important; }
h3 { font-size: 1.8rem !important; font-weight: 600 !important; color: var(--primary) !important; }
h4 { font-size: 1.5rem !important; font-weight: 600 !important; }

p, li, label, .stMarkdown, .stText, div, span {
    font-size: 20px !important;
    line-height: 1.7 !important;
    color: var(--text-main);
}

/* Cards & Containers */
.css-1r6slb0, .stTextArea, .stTextInput {
    border-radius: 12px !important;
}

/* Inputs: Force White Background & Black Text */
.stTextArea textarea, .stTextInput input, div[data-baseweb="textarea"], div[data-baseweb="input"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    caret-color: #000000; /* Black cursor */
    font-family: 'Inter', sans-serif !important;
    font-size: 22px !important; /* Even Bigger for readability */
    line-height: 1.6 !important;
    -webkit-text-fill-color: #000000 !important; /* Force override for Webkit */
}

/* Sidebar specific text color override */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {
    color: #0f172a !important; /* Dark Slate to contrast with white */
}

/* Fix st.info/st.success text inside sidebar */
[data-testid="stSidebar"] .stAlert div {
    color: #0f172a !important;
}

/* Force Button Text White */
.stButton button {
    color: #ffffff !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
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
    } 
    
    # Filter and add
    for m in regex_matches:
        if len(m) >= 3 and m not in STOPWORDS and m not in entities:
             entities.append(m)
             
    # DEBUG: Attach to session state if possible (hacky but needed)
    if "debug_regex" not in st.session_state: st.session_state.debug_regex = []
    st.session_state.debug_regex = regex_matches
    
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
        spaced_ent = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', ent)
        n_ent_spaced = normalize(spaced_ent)
        
        candidates = []
        # Check both the original run-together words and spaced words
        ent_words = set([w for w in n_ent.split() if len(w) >= 3] + 
                        [w for w in n_ent_spaced.split() if len(w) >= 3])
        
        for w in ent_words:
            if w in word_index:
                candidates.extend(word_index[w])
        
        # 3. Fuzzy Check on Candidates + Global Fuzzy fallback
        best_score = 0
        best_row = None
        
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
        if len(ent_first) >= 3: threshold = 0.45 
        
        if best_score > threshold: 
            forced[best_row["url"]] = {"row": best_row, "alias": ent}
            # TRACE LOG
            if "samphire" in n_ent:
                if "trace_log" not in st.session_state: st.session_state.trace_log = []
                st.session_state.trace_log.append(f"ACCEPTED '{ent}' -> {best_row['url']} (Score: {best_score:.2f} > {threshold})")
            continue
        else:
             if "samphire" in n_ent:
                if "trace_log" not in st.session_state: st.session_state.trace_log = []
                st.session_state.trace_log.append(f"REJECTED '{ent}' (Best: {best_score:.2f} <= {threshold}). Cands: {len(unique_cands)}")

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
            "strategy": """
            1. LINK ALL 'matched_via' CANDIDATES. These are pre-verified. Ignoring them is an error.
            2. ANCHOR TEXT MUST BE THE FULL NAME. 
               - BAD: <a...>Neuroscience</a>
               - GOOD: <a...>Flow Neuroscience</a>
               - BAD: <a...>Canyon</a> Energy
               - GOOD: <a...>Canyon Energy</a>
            3. Link "Canyon Energy" to "Canyon Magnet" if suggested.
            4. Link "Samphire" to "Samphire Neuroscience" if suggested.
            """,
            "output_format": [{"anchor": "FULL matched string from text", "url": "target url"}]
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
        u, a = ins.get('url'), str(ins.get('anchor', '')).strip()
        if not u or not a: continue
        if u in seen: continue
        seen.add(u)
        # Update anchor to stripped version
        ins['anchor'] = a
        valid_ins.append(ins)
    
    # Sort by anchor length (longest first) to match specific phrases before general ones
    valid_ins.sort(key=lambda x: len(x['anchor']), reverse=True)
    
    # 2. Split into tokens (tags vs text)
    tokens = re.split(r'(<[^>]+>)', html)
    
    # Store replacements map: placeholder -> (url, anchor)
    replacements = {}
    
    # Debug info
    applied_count = 0
    
    # 3. Perform replacement ONLY on text tokens
    for i, ins in enumerate(valid_ins):
        anchor = ins['anchor']
        url = ins['url']
        # Use a short, opaque placeholder
        pid = f"__L_{i}__" 
        replacements[pid] = f'<a href="{url}">{anchor}</a>'
        
        # Use simple escaping, verify regex validity
        safe_anchor = re.escape(anchor)
        # Ensure we don't match inside words if possible, but for now strict substring is safer for "Samphire"
        # We can add \b boundary if needed, but sometimes it breaks formatted text.
        pattern = re.compile(safe_anchor, re.IGNORECASE)
        
        found_for_this_link = False
        
        for k, token in enumerate(tokens):
            # Skip tags or placeholders
            if token.startswith('<') or token.startswith('__L_'):
                continue
                
            if found_for_this_link: break
            
            # Search
            match = pattern.search(token)
            if match:
                # Replace FIRST occurrence in this token
                # We use the matched text to preserve case if we wanted, but here we replace with placeholder
                new_token = token[:match.start()] + pid + token[match.end():]
                tokens[k] = new_token
                found_for_this_link = True
                applied_count += 1
                
        if not found_for_this_link:
             # DEBUG: Why was it missed?
             pass # st.toast(f"Missed link insertion for: {anchor}", icon="⚠️")

    # 4. Reassemble
    final_html = "".join(tokens)
    
    # 5. Swap placeholders for real links
    for pid, link_tag in replacements.items():
        final_html = final_html.replace(pid, link_tag)
        
    return final_html

def remove_link_from_html(html, url_to_remove):
    # Simplistic removal: replace <a href="TARGET">Anchor</a> with Anchor
    pattern = re.compile(rf'<a\s+href="{re.escape(url_to_remove)}"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    return pattern.sub(r'\\1', html)

def extract_links_for_ui(html):
    return [
        {"url": m.group(1), "anchor": m.group(2)}
        for m in re.finditer(r'<a href="([^"]+)">(.*?)</a>', html)
    ]

# ==================================================
# UI Layout
# ==================================================

def run_autolink_process(draft, mat, meta):
    with st.status("🚀 Processing...", expanded=True) as status:
        
        # 1. NER
        st.write("🔍 Identifying Key Entities (LLM)...")
        entities, usage_ner = identify_entities_with_llm(draft)
        st.write(f"   — Found: {', '.join(entities)}")
        
        # 2. Match to DB
        st.write("📂 Mapping to Knowledge Base...")
        forced_map = match_entities_to_db(entities, meta)
        
        # DEBUG: Show Samphire Trace
        if "trace_log" in st.session_state and st.session_state.trace_log:
            st.warning("🔎 Samphire Trace Log:")
            for log in st.session_state.trace_log:
                st.code(log, language="text")

        
        # 3. Candidates
        st.write("🧠 Building Semantic Candidates...")
        phrases = build_candidates_v2(draft, mat, meta, forced_map)
        
        # 4. Final Linking
        st.write("🔗 Generating Links...")
        resp, usage_link = call_llm_autolink(draft, phrases)
        
        insertions = resp.get("insertions", [])
        
        # 5. POST-PROCESS: Aggressive Merging Strategy
        # We have two sources of truth:
        # A. LLM Suggestions (resp.get("insertions"))
        # B. Our "Forced Map" (NER + Fuzzy matching) - effectively "Self-Generated" suggestions.
        
        # Strategy:
        # 1. Generate "Self Suggestions" from forced_map.
        # 2. Merge A + B.
        # 3. Collision Resolution: Exact same URL? Pick the one with the LONGEST Anchor.
        #    (e.g. "Flow Neuroscience" > "Neuroscience")
        
        candidates_map = {} # url -> insertion_dict
        
        # A. Process LLM Suggestions
        for ins in insertions:
            url = ins.get("url")
            anchor = ins.get("anchor")
            if not url or not anchor: continue
            
            # Verify existence (prevent hallucination)
            if re.search(re.escape(anchor), draft, re.IGNORECASE):
                candidates_map[url] = ins
        
        # DEBUG DATA
        merge_debug = []

        # B. Process Forced/Self Suggestions
        for url, data in forced_map.items():
            alias = data['alias']
            # We trust the alias exists because it came from NER/Regex, but let's be safe
            found_match = re.search(re.escape(alias), draft, re.IGNORECASE)
            
            merge_debug.append(f"Processing Forced Check: {alias} -> Found in text? {bool(found_match)}")

            if found_match:
                # Create a candidate
                self_ins = {"anchor": alias, "url": url, "source": "forced"}
                
                if url in candidates_map:
                    # Collision! Compare lengths.
                    existing_anchor = candidates_map[url]['anchor']
                    if len(alias) > len(existing_anchor):
                        candidates_map[url] = self_ins # Override with longer
                        st.write(f"⚠️ Override: Preferring '{alias}' over '{existing_anchor}'")
                else:
                    candidates_map[url] = self_ins
                    st.toast(f"Force-linked: {alias}", icon="⚡")
        
        final_insertions = list(candidates_map.values())
        
        if "trace_log" in st.session_state:
             with st.expander("🛠️ Debug: Linking Logic", expanded=True):
                 st.write("Merge Debug:", merge_debug)
                 st.write("Final Merged Candidates:", final_insertions)

        # 6. Apply
        final_html = apply_insertions(draft, final_insertions)
        
        # Cost Logic (unchanged)
        st.session_state.total_input_tokens += (usage_ner["prompt_tokens"] + usage_link["prompt_tokens"])
        st.session_state.total_output_tokens += (usage_ner["completion_tokens"] + usage_link["completion_tokens"])
        
        cost_in = (st.session_state.total_input_tokens / 1_000_000) * 0.30
        cost_out = (st.session_state.total_output_tokens / 1_000_000) * 2.50 
        st.session_state.total_cost = cost_in + cost_out

        st.session_state.result = {
            "html": final_html,
            "phrases": phrases,
            "forced_map": forced_map, 
            "entities": entities 
        }
        
        status.update(label="✅ Done!", state="complete", expanded=False)

def delete_link_callback(url_to_remove):
    if st.session_state.result:
        current_html = st.session_state.result["html"]
        new_html = remove_link_from_html(current_html, url_to_remove)
        st.session_state.result["html"] = new_html
        st.session_state.editor_content = new_html

def update_result_html():
    st.session_state.result["html"] = st.session_state.editor_content

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
            missed.sort(key=lambda x: x['score'] if x['score'] < 2 else 1.0, reverse=True) 
            
            st.info("These pages matched your content but weren't automatically linked. You can add them manually in the editor.")
            
            for m in missed[:15]: 
                score_display = "NER Match" if m.get("reason") != "Semantic Match" else f"Score: {m['score']:.2f}"
                st.markdown(f"**[{m['title']}]({m['url']})** — *{score_display}*")
                st.caption(f"Context: \"...{m['sentence'][:80]}...\"")
                st.markdown("---")
        
    # DEBUG SECTION (Temporary)
    with st.expander("🛠️ Debug: LLM Input Data"):
        st.write("Regex Raw Matches:", st.session_state.get("debug_regex", []))
        st.write("Trace Log (Samphire):", st.session_state.get("trace_log", []))
        if st.session_state.result:
            st.write("NER Entities:", st.session_state.result.get("entities", []))
            st.write("Matched Entities (Forced Map):", [f"{k} -> {v['row']['url']}" for k,v in st.session_state.result.get("forced_map", {}).items()])
            st.json(st.session_state.result.get("phrases", []))

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

if __name__ == "__main__":
    main()