import streamlit as st
import re
from dotenv import load_dotenv
import os

from styles import get_custom_css
from logic import (
    load_data, 
    identify_entities_with_llm, 
    match_entities_to_db, 
    build_candidates, 
    call_llm_autolink, 
    apply_insertions
)

# SETUP
load_dotenv()
st.set_page_config(
    page_title="SmartLink — AI Auto-Linker",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE
if "result" not in st.session_state: st.session_state.result = None
if "total_input_tokens" not in st.session_state: st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state: st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0

# INJECT CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# LOAD DATA
mat, meta = load_data()
if mat is None:
    st.error("Embeddings not found. Please check sosv_content/.")
    st.stop()

# SIDEBAR
with st.sidebar:
    st.markdown("## 📊 Stats") # Use H2/Markdown to avoid massive H1 style
    
    st.markdown(f"""
    <div style="background:#e0e7ff; padding:15px; border-radius:10px; border:1px solid #c7d2fe;">
        <h4 style="margin:0; font-size:16px; color:#4f46e5;">Cost: ${st.session_state.total_cost:.4f}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"In: {st.session_state.total_input_tokens}")
    st.caption(f"Out: {st.session_state.total_output_tokens}")
    
    if st.button("Reset Stats"):
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# MAIN UI
st.markdown('# 🔗 SmartLink')
st.markdown("Automated internal linking powered by **Gemini 2.5** and semantic search.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📝 Draft Content")
    draft_input = st.text_area("Paste text here...", height=300, label_visibility="collapsed")

with col2:
    st.markdown("### ⚙️ Controls")
    st.info(f"📚 Knowledge Base: {len(meta)} ent.")
    
    st.write("Link Types")
    st.radio("Link Types", ["Mixed", "Companies Only", "Content Only"], horizontal=True, label_visibility="collapsed")
    
    if st.button("✨ Auto-Link Draft", use_container_width=True):
        if not draft_input:
            st.warning("Please enter text.")
        else:
            with st.status("🚀 Processing...", expanded=True) as status:
                st.write("🔍 Extracting entities...")
                entities, usage_ner = identify_entities_with_llm(draft_input)
                st.caption(f"Found {len(entities)} raw entities.")
                
                st.write("📂 Mapping to Knowledge Base...")
                forced_map, trace_log = match_entities_to_db(entities, meta)
                
                # VISIBILITY: 1. Validated Candidates (Selected)
                validated_names = [d['alias'] for d in forced_map.values()]
                st.success(f"✅ Validated {len(validated_names)} companies: {', '.join(validated_names)}")
                
                # DEBUG TRACE -> Render as code block for readability
                with st.expander("Detailed Match Logic"):
                    st.code("\n".join(trace_log), language="text")
                
                st.write("🧠 Building Semantic Candidates...")
                phrases = build_candidates(draft_input, mat, meta, forced_map)
                
                st.write("🔗 Generating final links...")
                resp, usage_link = call_llm_autolink(draft_input, phrases)
                
                insertions = resp.get("insertions", [])
                
                # MERGE LOGIC (Aggressive Longest Match)
                candidates_map = {}
                
                # A. LLM
                for ins in insertions:
                    u, a = ins.get('url'), ins.get('anchor')
                    if u and a:
                        candidates_map[u] = ins
                
                # B. Forced
                for url, data in forced_map.items():
                    alias = data['alias']
                    # ROBUST MERGE CHECK:
                    found_match = False
                    if re.search(re.escape(alias), draft_input, re.IGNORECASE):
                        found_match = True
                    elif alias.lower() in draft_input.lower():
                        found_match = True
                        
                    if found_match:
                        self_ins = {"anchor": alias, "url": url, "source": "forced"}
                        if url in candidates_map:
                            if len(alias) > len(candidates_map[url]['anchor']):
                                candidates_map[url] = self_ins # Override
                        else:
                            candidates_map[url] = self_ins
                
                final_insertions = list(candidates_map.values())
                
                # VISIBILITY: 2. Final Links
                if final_insertions:
                    clean_list = [f"{i['anchor']} -> {i['url']}" for i in final_insertions]
                    st.info(f"🔗 Generatng {len(final_insertions)} Links:\n" + "\n".join(clean_list))
                else:
                    st.warning("No links generated.")

                st.write("📝 Applying changes...")
                final_html = apply_insertions(draft_input, final_insertions)
                
                # Cost Update
                t_in = usage_ner["prompt_tokens"] + usage_link["prompt_tokens"]
                t_out = usage_ner["completion_tokens"] + usage_link["completion_tokens"]
                st.session_state.total_input_tokens += t_in
                st.session_state.total_output_tokens += t_out
                st.session_state.total_cost += (t_in/1e6)*0.3 + (t_out/1e6)*2.5
                
                st.session_state.result = {
                    "html": final_html,
                    "phrases": phrases
                }
                
                status.update(label="✅ Done!", state="complete", expanded=False)
            
            # RERUN to update Sidebar Stats immediately
            st.rerun()

# SIDEBAR (MOVED TO BOTTOM OR UPDATED AFTER RERUN)
# Note: Streamlit runs script top-to-bottom. If we update stats at bottom, sidebar at top won't reflect until rerun.
# That is why we added st.rerun() above.

# RESULTS
if st.session_state.result:
    res = st.session_state.result
    
    st.markdown("### 🎉 Result")
    
    r_col1, r_col2 = st.columns([1, 1])
    
    with r_col1:
        st.markdown("#### ✏️ Editor (HTML)")
        st.caption("Edit HTML here to update preview.")
        html_val = st.text_area("Edit Code", value=res["html"], height=400)
        # Update state on edit
        if html_val != res["html"]:
            res["html"] = html_val
            st.rerun()

    with r_col2:
        st.markdown("#### 👁️ Live Preview")
        st.markdown(f'<div class="preview-box">{res["html"]}</div>', unsafe_allow_html=True)
