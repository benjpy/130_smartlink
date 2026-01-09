import re

def normalize(text: str) -> str:
    """Lowercases and keeps only alphanumeric and spaces. Replaces hyphens with spaces."""
    text = text.replace("-", " ") 
    return re.sub(r"[^a-z0-9 ]+", "", text.lower())

def split_sentences(text: str, max_sentences=40):
    """Splits text into chunks roughly by sentence."""
    # Quick heuristic split
    return [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
        if len(s.strip()) >= 20
    ][:max_sentences]

def entity_title_from_url(url: str) -> str:
    """Extracts a readable title from a URL slug."""
    return url.rstrip("/").split("/")[-1].replace("-", " ")
