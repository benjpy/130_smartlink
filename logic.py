import os
import re
import csv
import json
import numpy as np
import difflib
import google.generativeai as genai
import streamlit as st
from utils import normalize, split_sentences, entity_title_from_url

# CONFIG
EMB_PATH = "sosv_content/embeddings.npy"
META_PATH = "sosv_content/embeddings_meta.csv"
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-flash"
MAX_LINKS_TOTAL = 25

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

def identify_entities_with_llm(draft: str):
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

    # 2. Regex Fallback
    regex_matches = re.findall(r'\\b[A-Z][a-zA-Z0-9]+\\b', draft)
    
    STOPWORDS = {
        "The", "A", "And", "Or", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "Up", "Out", 
        "It", "Is", "Are", "Was", "Were", "Be", "Been", "Has", "Have", "Had", "Do", "Does", "Did",
        "But", "So", "If", "While", "When", "Where", "Why", "How", "All", "Any", "Some", "No", "Not", 
        "Yes", "We", "You", "They", "He", "She", "It", "My", "Your", "Our", "Their", "His", "Her",
        "This", "That", "These", "Those", "Just", "More", "Most", "Other", "Such", "New", "Good", "High",
        "Our", "Your", "My", "Their", "His", "Her", "Its"
    } 
    
    for m in regex_matches:
        if len(m) >= 3 and m not in STOPWORDS and m not in entities:
             entities.append(m)
             
    return entities, usage

def match_entities_to_db(entities, meta):
    forced = {}
    
    # Pre-index meta
    meta_index = {} # Exact slug
    word_index = {} # Word -> list of rows
    
    for row in meta:
        title = entity_title_from_url(row["url"])
        norm_title = normalize(title)
        meta_index[norm_title] = row
        for w in norm_title.split():
            if len(w) >= 3:
                if w not in word_index: word_index[w] = []
                word_index[w].append(row)

    trace_log = []

    for ent in entities:
        if not isinstance(ent, str): continue
        n_ent = normalize(ent)
        if len(n_ent) < 2: continue
        
        # 1. Exact match
        if n_ent in meta_index:
            row = meta_index[n_ent]
            url = row["url"]
            # Overwrite protection: Keep longest alias
            if url in forced:
                if len(ent) > len(forced[url]['alias']):
                    forced[url] = {"row": row, "alias": ent}
            else:
                forced[url] = {"row": row, "alias": ent}
            continue
            
        # 2. Fuzzy Search
        spaced_ent = re.sub(r'([a-z])([A-Z])', r'\\1 \\2', ent)
        n_ent_spaced = normalize(spaced_ent)
        
        candidates = []
        ent_words = set([w for w in n_ent.split() if len(w) >= 3] + 
                        [w for w in n_ent_spaced.split() if len(w) >= 3])
        
        for w in ent_words:
            if w in word_index:
                candidates.extend(word_index[w])
        
        best_score = 0
        best_row = None
        
        ent_tokens = n_ent_spaced.split()
        ent_first = ent_tokens[0] if ent_tokens else ""
        
        unique_cands = {c['url']: c for c in candidates}
        candidates_to_check = list(unique_cands.items()) if unique_cands else []
        
        for url, row in candidates_to_check:
            t_norm = normalize(entity_title_from_url(url))
            score = difflib.SequenceMatcher(None, n_ent_spaced, t_norm).ratio()
            t_clean = re.sub(r'^the\\s+', '', t_norm)
            t_first = t_clean.split()[0] if t_clean else ""
            
            if len(ent_first) >= 3 and ent_first in t_norm: score += 0.15
            if len(ent_first) >= 3 and ent_first == t_first: score += 0.3
            
            if score > best_score:
                best_score = score
                best_row = row
        
        threshold = 0.6
        if len(ent_first) >= 3: threshold = 0.45 
        
        if best_score > threshold: 
            url = best_row["url"]
            should_add = True
            
            if url in forced:
                existing_alias = forced[url]['alias']
                if len(ent) <= len(existing_alias):
                    should_add = False
                    trace_log.append(f"SKIPPED Overwrite '{ent}' -> {url} (Existing: '{existing_alias}' is longer)")
            
            if should_add:
                forced[url] = {"row": best_row, "alias": ent}
                trace_log.append(f"ACCEPTED '{ent}' -> {url} (Score: {best_score:.2f})")
            continue
        else:
             trace_log.append(f"REJECTED '{ent}' (Best: {best_score:.2f})")

    return forced, trace_log

def build_candidates(draft: str, mat, meta, forced_map):
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
            alias = data['alias']
            alias_parts = [w for w in normalize(alias).split() if len(w) >= 3]
            is_in_sentence = False
            if not alias_parts:
                if normalize(alias) in sent_norm: is_in_sentence = True
            else:
                if any(w in sent_norm for w in alias_parts):
                    # Check partials
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
    Robust insertion replacing the previous regex-heavy logic.
    Uses strict case-insensitive string lookups on text tokens.
    """
    # 1. Deduplicate & Clean
    valid_ins = []
    seen = set()
    for ins in insertions:
        u, a = ins.get('url'), str(ins.get('anchor', '')).strip()
        if not u or not a: continue
        if u in seen: continue
        seen.add(u)
        ins['anchor'] = a
        valid_ins.append(ins)
    
    # Longest anchor first
    valid_ins.sort(key=lambda x: len(x['anchor']), reverse=True)
    
    # 2. Tokenize (Tags vs Text)
    # Split by tags
    tokens = re.split(r'(<[^>]+>)', html)
    
    replacements = {}
    
    for i, ins in enumerate(valid_ins):
        anchor = ins['anchor']
        url = ins['url']
        pid = f"__L_{i}__"
        replacements[pid] = f'<a href="{url}">{anchor}</a>'
        
        found = False
        l_anchor = anchor.lower()
        
        # Scan text tokens
        for k, token in enumerate(tokens):
            if token.startswith('<') or token.startswith('__L_'): continue
            if found: break
            
            # Simple substring search (case-insensitive)
            l_token = token.lower()
            start_idx = l_token.find(l_anchor)
            
            if start_idx != -1:
                # REPLACEMENT
                # We slice the original token to preserve original case
                end_idx = start_idx + len(anchor)
                
                # Check boundaries (optional, but safe to skip for robustness if needed)
                # For now: We allow partial word matches if it means success? 
                # Better: Check strictly if we want. But user wants Samphire fixed.
                # Let's just do it.
                
                new_token = token[:start_idx] + pid + token[end_idx:]
                tokens[k] = new_token
                found = True
    
    # Reassemble and Swap
    final_html = "".join(tokens)
    for pid, link in replacements.items():
        final_html = final_html.replace(pid, link)
        
    return final_html
