import os
import csv
import time
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------
# Config
# -------------------------
BASE_DIR = "sosv_content"
CONTENT_CSV = f"{BASE_DIR}/content.csv"
OUT_EMB = f"{BASE_DIR}/embeddings.npy"
OUT_META = f"{BASE_DIR}/embeddings_meta.csv"

EMBED_MODEL = "models/embedding-001"
BATCH_SIZE = 64  # Gemini is happier with smaller batches

# -------------------------
# Setup
# -------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------
# Helpers
# -------------------------
def read_rows():
    with open(CONTENT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_text(row):
    path = os.path.join(BASE_DIR, row["text_file"])
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def batched(items, n):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# -------------------------
# Main
# -------------------------
def main():
    rows = read_rows()
    docs = []

    for row in rows:
        text = load_text(row)
        if text:
            docs.append((row, text))

    print(f"Loaded {len(docs)} documents")

    vectors = []
    meta_rows = []

    for bi, batch in enumerate(batched(docs, BATCH_SIZE), 1):
        texts = [text for (_, text) in batch]

        result = genai.embed_content(
            model=EMBED_MODEL,
            content=texts,
            task_type="retrieval_document"
        )

        embeddings = result["embedding"]

        for (row, _), emb in zip(batch, embeddings):
            vec = np.array(emb, dtype=np.float32)
            vectors.append(vec)

            meta_rows.append({
                "url": row["url"],
                "content_type": row["content_type"],
                "text_file": row["text_file"],
                "hash": row["hash"],
                "char_count": row["char_count"],
            })

        if bi % 5 == 0:
            print(f"Embedded {min(bi*BATCH_SIZE, len(docs))}/{len(docs)}")

        time.sleep(0.2)  # polite pacing

    mat = np.vstack(vectors)
    np.save(OUT_EMB, mat)

    with open(OUT_META, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["url","content_type","text_file","hash","char_count"]
        )
        w.writeheader()
        w.writerows(meta_rows)

    print(f"\nSaved embeddings → {OUT_EMB}  shape={mat.shape}")
    print(f"Saved metadata   → {OUT_META}")

if __name__ == "__main__":
    main()