import json
import faiss
import numpy as np
import psycopg2
from tqdm import tqdm

# --- Config ---
MODEL_DIM = 384
FAISS_INDEX_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/faiss_index.index"
METADATA_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/metadata.json"

DB_CONFIG = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}

# --- Connect to DB ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# --- Get total count first ---
cur.execute("SELECT COUNT(*) FROM papers p JOIN embeddings e ON p.id = e.paper_id;")
total_rows = cur.fetchone()[0]

# --- Stream data with progress ---
print(f"Fetching {total_rows} embeddings and metadata from PostgreSQL...")
cur.execute("""
    SELECT 
    p.id, 
    p.title, 
    COALESCE(vn.normalized_venue, p.venue) AS final_venue, 
    p.n_citation, 
    e.abstract_embedding
    FROM papers p
    JOIN embeddings e ON p.id = e.paper_id
    LEFT JOIN venues_norm vn ON p.venue = vn.raw_venue;
""")

embeddings = []
metadata = []

with tqdm(total=total_rows, desc="Processing Papers") as pbar:
    while True:
        row = cur.fetchone()
        if row is None:
            break

        paper_id, title, venue, n_citation, embedding_pg = row
        vec = np.array(json.loads(embedding_pg), dtype=np.float32)

        if vec.shape[0] != MODEL_DIM:
            pbar.update(1)
            continue  # Skip corrupted

        embeddings.append(vec)
        metadata.append({
            "id": paper_id,
            "title": title,
            "venue": venue,
            "n_citation": n_citation
        })

        pbar.update(1)

# --- Build FAISS index ---
print("Building FAISS index...")
faiss_index = faiss.IndexFlatIP(MODEL_DIM)
faiss_index.add(np.array(embeddings))

# --- Save files ---
faiss.write_index(faiss_index, FAISS_INDEX_PATH)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved FAISS index to: {FAISS_INDEX_PATH}")
print(f"Saved metadata to: {METADATA_PATH}")

cur.close()
conn.close()
