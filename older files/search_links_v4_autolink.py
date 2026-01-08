import csv
import os
import re
import json
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
    page_title="Internal Linking Assistant — LLM Auto-Link",
    layout="wide"
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
# CSS — FORCE LIGHT THEME
# ==================================================
st.markdown("""
<style>
html, body, .stApp {
    background:#ffffff !important;
    color:#111111 !important;
    font-size:22px !important;
}
.block-container {
    max-width:1500px;
    padding:3rem 0 4rem;
}
div[data-baseweb="textarea"] textarea,
div[data-baseweb="textarea"] * {
    background:#ffffff !important;
    color:#111111 !important;
    font-size:20px !important;
    line-height:1.75 !important;
}
div.stButton > button {
    background:#2563eb !important;
    color:#ffffff !important;
    font-size:18px !important;
    font-weight:600 !important;
    border-radius:14px !important;
    border:none !important;
}
div.stButton > button:hover {
    background:#1e40af !important;
}
button.remove-btn {
    background:#ef4444 !important;
}
.helper {
    font-size:17px;
    color:#6b7280;
}
button[data-baseweb="tab"] div p,
button[data-baseweb="tab"] p {
</style>
""", unsafe_allow_html=True)

# ==================================================
# Helpers
# ==================================================
@st.cache_resource
def load_embeddings_v2():
    mat = np.load(EMB_PATH).astype(np.float32)
    mat = np.nan_to_num(mat, copy=False)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    mat = np.divide(mat, norm, out=np.zeros_like(mat), where=norm!=0)
    with open(META_PATH, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return mat, meta

def batch_embed(texts: list) -> np.ndarray:
    if not texts:
        return np.array([])
    # Handle batch embedding
    # Note: embed_content can take a list. 
    # Check your library version if 'content' vs 'texts' is needed, but 'content' usually works for list.
    resp = genai.embed_content(model=EMBED_MODEL, content=texts)
    embeddings = resp["embedding"]
    
    mat = np.array(embeddings, dtype=np.float32)
    mat = np.nan_to_num(mat, copy=False)
    
    # Safe normalization
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    # Replace zero norms with 1 to avoid division by zero (result remains 0 vector)
    norm[norm == 0] = 1.0
    mat = mat / norm
    
    return mat

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())

def split_sentences(text: str):
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) >= 30
    ][:MAX_SENTENCES]

def entity_title_from_url(url: str):
    return url.rstrip("/").split("/")[-1].replace("-", " ")

# ==================================================
# Deterministic company auto-linking
# ==================================================
def auto_link_companies(draft: str, meta):
    out = draft
    for row in meta:
        if row.get("content_type") != "company":
            continue
        name = entity_title_from_url(row["url"])
        pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        m = pat.search(out)
        if not m:
            continue
        if out[:m.start()].count("<a ") > out[:m.start()].count("</a>"):
            continue
        anchor = out[m.start():m.end()]
        out = out[:m.start()] + f'<a href="{row["url"]}">{anchor}</a>' + out[m.end():]
    return out

def extract_links_from_html(html: str):
    return [
        {"url": m.group(1), "anchor": m.group(2)}
        for m in re.finditer(r'<a href="([^"]+)">(.*?)</a>', html)
    ]

def remove_link(html: str, url: str, anchor: str):
    return re.sub(
        re.escape(f'<a href="{url}">{anchor}</a>'),
        anchor,
        html,
        count=1
    )

def format_url_display(url: str) -> str:
    if "/company/" in url:
        slug = url.split("/company/")[-1]
        if slug.endswith("/"): slug = slug[:-1]
        return f"**[Company]** {slug}"
    
    # Fallback / News
    parts = url.strip("/").split("/")
    slug = parts[-1] if parts else url
    return f"**[News]** {slug}"


# ==================================================
# Candidate building (with forced entity matches)
# ==================================================
def force_explicit_entity_matches(sentence: str, meta):
    sent_norm = normalize(sentence)
    forced = {}
    
    # Common stopwords to ignore in matching
    STOPWORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
        "this", "but", "his", "by", "from", "they", "we", "say", "her", 
        "she", "or", "an", "will", "my", "one", "all", "would", "there", 
        "their", "what", "so", "up", "out", "if", "about", "who", "get", 
        "which", "go", "me", "when", "make", "can", "like", "time", "no", 
        "just", "him", "know", "take", "people", "into", "year", "your", 
        "good", "some", "could", "them", "see", "other", "than", "then", 
        "now", "look", "only", "come", "its", "over", "think", "also", 
        "back", "after", "use", "two", "how", "our", "work", "first", 
        "well", "way", "even", "new", "want", "because", "any", "these", 
        "give", "day", "most", "us"
    }

    for row in meta:
        title = entity_title_from_url(row["url"])
        title_norm = normalize(title)
        
        # Split into tokens & filter stopwords
        t_tokens = set(w for w in title_norm.split() if w not in STOPWORDS)
        s_tokens = set(w for w in sent_norm.split() if w not in STOPWORDS)
        shared = t_tokens & s_tokens
        
        match = False
        # Rule 1: 2+ shared words (strong match)
        if len(shared) >= 2:
            match = True
        # Rule 2: 1 shared word, but it's a "unique" name (heuristic)
        # Word must be >= 5 chars (avoids 'fund', 'team', 'flow')
        # Entity title must be short (<= 3 words) to avoid matching long sentences
        elif len(shared) == 1:
            w = list(shared)[0]
            if len(w) >= 5 and len(t_tokens) <= 3:
                match = True
                
        if match:
            forced[row["url"]] = {
                "url": row["url"],
                "title": title,
                "type": row.get("content_type", ""),
                "score": 1.0,
                "is_strong_match": True
            }
    return forced

def build_candidates(draft: str, mat, meta):
    sentences = split_sentences(draft)
    items = []
    
    if not sentences:
        return []

    # Batch embed all sentences
    # Returns (N_sentences, D_embedding)
    try:
        sent_embeddings = batch_embed(sentences)
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return []

    # Vectorized scoring: (N_docs, D) @ (N_sents, D).T -> (N_docs, N_sents)
    # Transpose result to get (N_sents, N_docs)
    # Pre-calculate indices by type to speed up filtering if needed, 
    # but iterating sorted results is usually fine for this dataset size.
    all_scores = (mat @ sent_embeddings.T).T 
    
    LIMIT_PER_TYPE = 10

    for i, sent in enumerate(sentences):
        pid = i + 1
        scores = all_scores[i]  # Scores for this sentence against all docs
        
        # 1. Force explicitly matched entities
        forced = force_explicit_entity_matches(sent, meta)
        
        # 2. Get top candidates globally -> filter into buckets
        top_idx = np.argsort(-scores)[:500]
        
        buckets = {"company": [], "event": [], "post": []}
        
        for idx in top_idx:
            row = meta[idx]
            ctype = row.get("content_type", "post")
            if ctype not in buckets: ctype = "post"
            
            if len(buckets[ctype]) < LIMIT_PER_TYPE:
                buckets[ctype].append({
                    "url": row["url"],
                    "title": entity_title_from_url(row["url"]),
                    "type": row.get("content_type", ""),
                    "score": float(scores[idx]),
                    "is_strong_match": False
                })
            if all(len(b) >= LIMIT_PER_TYPE for b in buckets.values()):
                break

        # 3. Merge
        final_candidates = {}
        for url, obj in forced.items():
            final_candidates[url] = obj
            
        for ctype in ["company", "event", "post"]:
            for cand in buckets[ctype]:
                if cand["url"] not in final_candidates:
                    final_candidates[cand["url"]] = cand
        
        items.append({
            "phrase_id": pid,
            "sentence": sent,
            "candidates": list(final_candidates.values())
        })

    return items

# ==================================================
# LLM call (NORMALIZED OUTPUT)
# ==================================================
def call_llm_autolink(draft: str, phrases: list, mode: str = "mixed"):
    # mode: 'mixed', 'companies_only', 'content_only'
    
    # Filter candidates based on mode
    filtered_phrases = []
    for p in phrases:
        s_candidates = []
        for c in p["candidates"]:
            ctype = c.get("type", "post")
            if mode == "companies_only":
                if ctype == "company":
                    s_candidates.append(c)
            elif mode == "content_only":
                if ctype != "company":
                    s_candidates.append(c)
            else:
                s_candidates.append(c)
        
        if s_candidates:
            filtered_phrases.append({
                "phrase_id": p["phrase_id"],
                "sentence": p["sentence"],
                "candidates": s_candidates
            })

    if not filtered_phrases:
        return {"insertions": []}, {"prompt_tokens": 0, "completion_tokens": 0}

    model = genai.GenerativeModel(LLM_MODEL)

    instruction = "Link entities to their URLs."
    if mode == "companies_only":
        instruction = "Link ONLY Companies. Ignore interactions, just identify company names and link them."
    elif mode == "content_only":
        instruction = "Link ONLY Events and Posts (News, Blogs). Do NOT link Companies."
    
    # Build Prompt
    payload = {
        "draft": draft,
        "phrases": filtered_phrases,
        "rules": {
            "max_links": MAX_LINKS_TOTAL,
            "no_duplicate_urls": True,
            "use_exact_anchor_text": False,
            "linking_strategy": instruction
        }
    }

    resp = model.generate_content(
        json.dumps(payload),
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json"
        )
    )

    raw = json.loads(resp.text)


    
    usage = {
        "prompt_tokens": resp.usage_metadata.prompt_token_count,
        "completion_tokens": resp.usage_metadata.candidates_token_count
    }

    # ✅ NORMALIZE GEMINI OUTPUT
    if isinstance(raw, list):
        return {"insertions": raw}, usage
    if isinstance(raw, dict):
        return raw, usage

    raise ValueError("Unexpected LLM output format")

# ==================================================
# UI
# ==================================================
st.markdown("## 🔗 Internal Linking Assistant — LLM Auto-Link")

draft = st.text_area("Draft", height=420)

st.markdown(
    '<div class="helper">Uses Gemini 2.5 Flash to select links and anchors.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

if col1.button("Auto-link with Gemini"):
    if not draft.strip():
        st.warning("Paste a draft first.")
    else:
        with st.status("Starting Auto-Link Process...", expanded=True) as status:
            import time
            
            st.write("💾 Loading knowledge base...")
            time.sleep(0.8)
            mat, meta = load_embeddings_v2()
            
            st.write("🕵️‍♀️ Scanning for direct matches...")
            time.sleep(0.8)
            html = auto_link_companies(draft, meta)
            
            st.write("🧠 Analyzing semantic context & building candidates...")
            time.sleep(0.8)
            phrases = build_candidates(html, mat, meta)
            candidate_count = sum(len(p['candidates']) for p in phrases)
            st.write(f"   — Found {candidate_count} potential links across {len(phrases)} segments.")
            

            
            st.write("🔮 Phase 1: Linking Companies (Prioritized)...")
            time.sleep(0.5)
            resp_c, usage_c = call_llm_autolink(html, phrases, mode="companies_only")
            
            st.write("🔮 Phase 2: Linking Events & Posts (Contextual)...")
            time.sleep(0.5)
            resp_e, usage_e = call_llm_autolink(html, phrases, mode="content_only")
            
            # Merge usages
            usage = {
                "prompt_tokens": usage_c["prompt_tokens"] + usage_e["prompt_tokens"],
                "completion_tokens": usage_c["completion_tokens"] + usage_e["completion_tokens"]
            }
            
            # Merge insertions
            all_insertions = resp_c.get("insertions", []) + resp_e.get("insertions", [])
            llm_resp = {"insertions": all_insertions}
            
            # Update Cost (Gemini 2.5 Flash Pricing as proxy for 2.5)
            # Input: $0.3 / 1M tokens
            # Output: $2.50 / 1M tokens
            cost_input = (usage["prompt_tokens"] / 1_000_000) * 0.3
            cost_output = (usage["completion_tokens"] / 1_000_000) * 2.50
            total_cost = cost_input + cost_output
            
            st.session_state.total_input_tokens += usage["prompt_tokens"]
            st.session_state.total_output_tokens += usage["completion_tokens"]
            st.session_state.total_cost += total_cost
            
            st.write("✨ Finalizing link insertions...")
            for ins in llm_resp.get("insertions", []):
                anchor = ins.get("anchor", "").strip()
                url = ins.get("url", "").strip()
                if not anchor or not url:
                    continue
                m = re.search(re.escape(anchor), html)
                if not m:
                    continue
                if html[:m.start()].count("<a ") > html[:m.start()].count("</a>"):
                    continue
                html = html[:m.start()] + f'<a href="{url}">{anchor}</a>' + html[m.end():]
            
            status.update(label="✅ Auto-linking Complete!", state="complete", expanded=False)

            st.session_state.result = {"html": html, "phrases": phrases}

# Sidebar Cost Tracker
with st.sidebar:

    st.header("💰 Session Cost")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    st.text(f"Input Tokens: {st.session_state.total_input_tokens:,}")
    st.text(f"Output Tokens: {st.session_state.total_output_tokens:,}")
    if st.button("Reset Cost"):
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

if col2.button("Clear"):
    st.session_state.result = None
    st.rerun()

# ==================================================
# Output
# ==================================================
if st.session_state.result:
    html = st.session_state.result["html"]
    links = extract_links_from_html(html)

    st.markdown("### Output")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        tab_preview, tab_html = st.tabs(["Rich Text Preview", "HTML Source"])
        with tab_preview:
            st.markdown("**Preview (Copyable)**")
            st.markdown(html, unsafe_allow_html=True)
        with tab_html:
            st.markdown("**Final HTML (copy/paste into WordPress)**")
            st.text_area("HTML Output", html, height=420, label_visibility="collapsed")

    with c2:
        st.markdown("**Selected links**")
        for i, l in enumerate(links):
            cols = st.columns([0.15, 0.85])
            with cols[0]:
                if st.button("❌", key=f"rm_{i}", help="Remove link"):
                    html = remove_link(html, l["url"], l["anchor"])
                    st.session_state.result["html"] = html
                    st.rerun()
            with cols[1]:
                display_text = format_url_display(l["url"])
                st.markdown(f"{l['anchor']} → {display_text}")

    # Debug Section
    with st.expander("🛠️ Debug Info: Link Candidates"):
        st.write("These are the candidates sent to the LLM for each sentence:")
        phrases = st.session_state.result.get("phrases", [])
        
        debug_text = ""
        for p in phrases:
            debug_text += f"\nSentence: {p['sentence']}\n"
            for c in p['candidates']:
                score_str = "FORCE" if c['score'] == 1.0 else f"{c['score']:.2f}"
                strong_mark = "💪" if c.get('is_strong_match') else "  "
                debug_text += f"  [{score_str}] {strong_mark} {c['type'].upper()} - {c['title']} ({c['url']})\n"
            debug_text += "-" * 40 + "\n"
        
        st.code(debug_text, language="text")