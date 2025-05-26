import json
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer


DB_CONFIG = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}

MODEL_NAME = "thenlper/gte-small"
OUTPUT_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/canonical_venue_embeddings.json"

#Embedder
embedder = SentenceTransformer(MODEL_NAME)

#Connect to DB
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

#Fetch distinct normalized venue names
cur.execute("""
    SELECT DISTINCT LOWER(TRIM(normalized_venue))
    FROM venues_norm
    WHERE normalized_venue IS NOT NULL AND normalized_venue <> '';
""")
venues = [row[0] for row in cur.fetchall()]
print(f"Fetched {len(venues)} distinct canonical venue names from venues_norm.")


#Embed each venue
canonical_venues = []
batch_size = 256

for i in range(0, len(venues), batch_size):
    batch = venues[i:i+batch_size]
    embeddings = embedder.encode(batch, normalize_embeddings=True)
    for name, vec in zip(batch, embeddings):
        canonical_venues.append({
            "name": name,
            "embedding": vec.tolist()
        })


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(canonical_venues, f, indent=2, ensure_ascii=False)


cur.close()
conn.close()

print(f"\nSaved embeddings to: {OUTPUT_PATH}")
