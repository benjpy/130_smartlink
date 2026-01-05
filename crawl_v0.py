import csv
import hashlib
import os
import time
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

def crawl():
    rows = []

    for content_type, sitemap_url in SITEMAPS.items():
        print(f"\nCrawling {content_type} sitemap...")
        urls = parse_sitemap(sitemap_url)

        for i, url in enumerate(urls, 1):
            try:
                html = fetch(url)
                text = extract_text(html)

                # Events can be shorter; posts/companies usually longer
                min_len = 250 if content_type == "event" else 350
                if len(text) < min_len:
                    continue

                fname = safe_filename(content_type, url)
                path = os.path.join(TXT_DIR, fname)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

                rows.append({
                    "url": url,
                    "content_type": content_type,
                    "text_file": f"txt/{fname}",
                    "char_count": len(text),
                    "hash": sha256(text),
                })

                if i % 20 == 0:
                    print(f"{content_type}: {i}/{len(urls)} processed")

                time.sleep(0.2)

            except Exception as e:
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