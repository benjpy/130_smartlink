
def get_custom_css():
    return """
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

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 20px -3px rgba(79, 70, 229, 0.4) !important;
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Status & Expander */
div[data-testid="stStatusWidget"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
}

/* FORCE all text inside Status Widget to be dark (fixes white-on-white in Dark Mode) */
div[data-testid="stStatusWidget"] div, 
div[data-testid="stStatusWidget"] span, 
div[data-testid="stStatusWidget"] label,
div[data-testid="stStatusWidget"] p {
    color: #0f172a !important;
}

.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #0f172a !important;
    background-color: #f1f5f9 !important; /* Slight grey for contrast */
    border-radius: 8px;
    font-size: 16px !important;
    border: 1px solid #e2e8f0;
}

.streamlit-expanderContent {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0; /* border to match header */
    border-top: none;
}

/* Code Blocks (Trace Logs) */
code {
    color: #d63384 !important; /* Pink text */
    background-color: #f8f9fa !important; /* Light grey bg */
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: bold;
}

/* Radio Buttons */
div[role="radiogroup"] label {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 10px 15px;
    border-radius: 8px;
    margin-right: 10px;
    transition: all 0.2s;
    font-size: 16px !important; /* Smaller than body for compactness */
}

div[role="radiogroup"] label:hover {
    border-color: var(--primary);
    background-color: #eef2ff;
}

/* General Text Overrides to kill Dark Mode leakage */
p, li, label, .stMarkdown, .stText, div, span, h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important; /* Force Dark Slate */
}

/* Success/Error/Info Alerts */
div[data-baseweb="notification"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    border-radius: 10px;
}

.preview-box a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 600;
    border-bottom: 2px solid rgba(79, 70, 229, 0.2);
    transition: all 0.2s;
}
.preview-box a:hover {
    background-color: rgba(79, 70, 229, 0.05);
    border-bottom-color: var(--primary);
}

/* Utility */
.stToast {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 15px 25px -5px rgba(0,0,0,0.1) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif;
}
</style>
"""
