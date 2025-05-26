import psycopg2
import json
import ijson
import os

# --- Config ---
DB_CONFIG = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}

JSON_PATH = "C:/Users/effik/Downloads/dblp_v14/dblp_v14.json"
OUTPUT_FOLDER = "C:/Users/effik/Desktop/THESIS/test_postgre/Data"
NUM_SETS = 10
SET_SIZE = 100

#Load DB Data
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

used_paper_ids = set()
all_sets = []

print(f"Fetching {NUM_SETS} × {SET_SIZE} unique papers from DB...")

for set_idx in range(NUM_SETS):
    paper_map = {}
    attempts = 0
    while len(paper_map) < SET_SIZE and attempts < 50:
        cur.execute("""
            SELECT p.id, COALESCE(vn.normalized_venue, p.venue) AS final_venue
            FROM papers p
            LEFT JOIN venues_norm vn ON p.venue = vn.raw_venue
            WHERE p.venue IS NOT NULL
            ORDER BY random()
            LIMIT %s;
        """, (SET_SIZE * 2,))
        rows = cur.fetchall()

        for paper_id, venue in rows:
            if paper_id not in used_paper_ids and venue and paper_id not in paper_map:
                paper_map[paper_id] = venue
                used_paper_ids.add(paper_id)
                if len(paper_map) >= SET_SIZE:
                    break

        attempts += 1

    all_sets.append(paper_map)

conn.close()

#Stream JSON and Extract Abstracts
print("\nStreaming json file to match abstracts...")

all_needed_ids = set(pid for s in all_sets for pid in s)
paper_abstract_map = {}

with open(JSON_PATH, "r", encoding="utf-8") as f:
    parser = ijson.items(f, "item")
    for paper in parser:
        paper_id = paper.get("id")
        if paper_id in all_needed_ids:
            abstract = paper.get("abstract", "").strip()
            if abstract:
                paper_abstract_map[paper_id] = abstract
        if len(paper_abstract_map) >= len(all_needed_ids):
            break

#Save Evaluation Sets
for i, paper_map in enumerate(all_sets, start=1):
    evaluation_set = []
    missing = 0
    for pid, venue in paper_map.items():
        abstract = paper_abstract_map.get(pid)
        if abstract:
            evaluation_set.append({
                "abstract": abstract,
                "actual_venue": venue
            })
        else:
            missing += 1

    output_path = os.path.join(OUTPUT_FOLDER, f"evaluation_set{i}.json")
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(evaluation_set, out, indent=2, ensure_ascii=False)

    print(f"\nSet {i}: Saved {len(evaluation_set)} papers to {output_path}")
    print(f"Missing abstracts for {missing} papers in Set {i}.")

print("\nEvaluation sets are ready.")
