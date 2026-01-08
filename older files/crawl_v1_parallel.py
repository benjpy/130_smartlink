import csv
import hashlib
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_WORKERS = 8  # 👈 safe range: 5–10

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def extract_text(html):
    text = trafilatura.extract(html, include_tables=False, include_comments=False)
    return (text or "").strip()

def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def parse_sitemap(url):
    xml = fetch(url)
    soup = BeautifulSoup(xml, "xml")
    return [loc.text.strip() for loc in soup.find_all("loc")]

def safe_filename(content_type, url):
    slug = url.rstrip("/").split("/")[-1]
    return f"{content_type}__{slug}.txt"

def process_url(content_type, url):
    html = fetch(url)
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

def crawl():
    rows = []

    for content_type, sitemap_url in SITEMAPS.items():
        print(f"\nCrawling {content_type} sitemap...")
        urls = parse_sitemap(sitemap_url)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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