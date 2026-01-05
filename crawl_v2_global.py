import csv
import hashlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import trafilatura

HEADERS = {"User-Agent": "SOSV-Linking-MVP/0.1"}

OUT_DIR = "sosv_content"
TXT_DIR = os.path.join(OUT_DIR, "txt")
os.makedirs(TXT_DIR, exist_ok=True)

SITEMAPS = {
    "post": "https://sosv.com/post-sitemap.xml",
    "company": "https://sosv.com/company-sitemap.xml",
    "event": "https://sosv.com/event-sitemap.xml",
}

# 👇 Different limits per content type (important)
WORKERS = {
    "post": 10,
    "company": 4,   # stricter rate limits
    "event": 6,
}

session = requests.Session()
session.headers.update(HEADERS)

# -------------------------
# Networking with backoff
# -------------------------
def fetch_with_backoff(url, session, max_retries=5):
    delay = 1.0

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            r.raise_for_status()
            return r.text

        except Exception:
            if attempt == max_retries - 1:
                raise

            sleep_time = delay + random.uniform(0, 0.5)
            time.sleep(sleep_time)
            delay *= 2

# -------------------------
# Parsing helpers
# -------------------------
def extract_text(html):
    return (trafilatura.extract(
        html,
        include_tables=False,
        include_comments=False,
        include_links=False,
        favor_recall=False,
    ) or "").strip()

def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_sitemap(url):
    xml = fetch_with_backoff(url, session)
    soup = BeautifulSoup(xml, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]

def safe_filename(content_type, url):
    slug = url.rstrip("/").split("/")[-1]
    return f"{content_type}__{slug}.txt"

# -------------------------
# Worker task
# -------------------------
def process_url(content_type, url):
    # small jitter to avoid synchronized bursts
    time.sleep(random.uniform(0.05, 0.15))

    html = fetch_with_backoff(url, session)

    # fast reject for tiny pages
    if len(html) < 3000:
        return None

    text = extract_text(html)

    min_len = 250 if content_type == "event" else 350
    if len(text) < min_len:
        return None

    fname = safe_filename(content_type, url)
    path = os.path.join(TXT_DIR, fname)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "url": url,
        "content_type": content_type,
        "text_file": f"txt/{fname}",
        "char_count": len(text),
        "hash": sha256(text),
    }

# -------------------------
# Main crawl
# -------------------------
def crawl():
    rows = []

    for content_type, sitemap_url in SITEMAPS.items():
        print(f"\nCrawling {content_type} sitemap...")
        urls = parse_sitemap(sitemap_url)

        max_workers = WORKERS[content_type]
        print(f"Using {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_url, content_type, url): url
                for url in urls
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    if result:
                        rows.append(result)

                    if i % 20 == 0:
                        print(f"{content_type}: {i}/{len(urls)} processed")

                except Exception as e:
                    url = futures[future]
                    print(f"Failed {url}: {e}")

    with open(os.path.join(OUT_DIR, "content.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["url", "content_type", "text_file", "char_count", "hash"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Indexed {len(rows)} items total.")

if __name__ == "__main__":
    crawl()