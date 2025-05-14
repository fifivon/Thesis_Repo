import psycopg2
import json
import ijson  # streaming JSON parser

# --- Config ---
DB_CONFIG = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}

JSON_PATH = "C:/Users/effik/Downloads/dblp_v14/dblp_v14.json"  # <- your file path for retrieving the og abstracts
#OUTPUT_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/evaluation_set.json"  # the one with the 1000 not-random samples (i think)
OUTPUT_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/evaluation_set2.json"  # the one with (currently) 100 random samples
TOP_N = 100

# --- Step 1: Fetch paper IDs and venues from the DB ---
print("Connecting to database...")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
    SELECT p.id, COALESCE(vn.normalized_venue, p.venue) AS final_venue
    FROM papers p
    LEFT JOIN venues_norm vn ON p.venue = vn.raw_venue
    WHERE p.venue IS NOT NULL
    ORDER BY random()
    LIMIT %s;
""", (TOP_N,))

rows = cur.fetchall()
conn.close()

# Build lookup: paper_id -> venue
paper_id_to_venue = {row[0]: row[1] for row in rows}
paper_ids_needed = set(paper_id_to_venue.keys())

print("\nPaper IDs we're trying to match from the DB:\n")
for pid in paper_id_to_venue:
    print(pid)

print(f"\nGot {len(paper_ids_needed)} paper IDs from the database.")

# --- Step 2: Parse JSON array file using ijson ---
print("Parsing JSON to find matching abstracts...")

evaluation_set = []
found_ids = set()
missing_abstracts = 0

with open(JSON_PATH, "r", encoding="utf-8") as f:
    parser = ijson.items(f, "item")  # each item in top-level array

    for paper in parser:
        paper_id = paper.get("id")
        if paper_id in paper_ids_needed:
            found_ids.add(paper_id)
            abstract = paper.get("abstract", "").strip()
            if abstract:
                evaluation_set.append({
                    "abstract": abstract,
                    "actual_venue": paper_id_to_venue[paper_id]
                })
            else:
                print(f"Found paper {paper_id} but abstract is missing.")
                missing_abstracts += 1

            if len(evaluation_set) >= TOP_N:
                break

# --- Step 3: Save the evaluation set ---
print(f"\nFinished parsing.")
print(f"Matched paper IDs in JSON: {len(found_ids)} / {len(paper_ids_needed)}")
print(f"Abstracts found: {len(evaluation_set)}")
print(f"Missing abstracts for matched IDs: {missing_abstracts}")

print(f"\nSaving {len(evaluation_set)} entries to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    json.dump(evaluation_set, out, indent=2, ensure_ascii=False)

print("Evaluation set is ready.")
