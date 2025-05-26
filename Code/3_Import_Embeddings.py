import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import ijson
from tqdm import tqdm
import torch


JSON_PATH = r"C:/Users/effik/Downloads/dblp_v14/dblp_v14.json"
DB_CONFIG = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}
MODEL_NAME = "thenlper/gte-small"
BATCH_SIZE = 64

#Load model with CUDA
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    model = model.to(torch.device("cuda"))
else:
    print("CUDA not available. Using CPU.")

# Paper info from DB
def load_paper_data_from_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, keywords FROM papers
        WHERE title IS NOT NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"Loaded {len(rows)} papers from DB.")
    return {pid: (title, keywords) for pid, title, keywords in rows}

#Stream abstracts from JSON file
def merge_with_abstracts(json_path, db_papers):
    with open(json_path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')
        for paper in papers:
            try:
                pid = paper.get('id')
                if pid in db_papers:
                    abstract = paper.get('abstract')
                    if abstract:
                        title, keywords = db_papers[pid]
                        yield (pid, title, abstract, keywords)
            except Exception as e:
                print(f"Error processing paper: {e}")

#Process a batch of papers
def process_batch(batch):
    pids, titles, abstracts, keywords = zip(*batch)
    kw_strings = [" ".join(kws) if kws else "" for kws in keywords]

    title_embs = model.encode(titles, batch_size=len(titles)).tolist()
    abstract_embs = model.encode(abstracts, batch_size=len(abstracts)).tolist()
    kw_embs = model.encode(kw_strings, batch_size=len(kw_strings)).tolist()

    for pid, t_emb, a_emb, k_emb in zip(pids, title_embs, abstract_embs, kw_embs):
        yield (
            pid,
            f"[{','.join(map(str, t_emb))}]",
            f"[{','.join(map(str, a_emb))}]",
            f"[{','.join(map(str, k_emb))}]"
        )

#Generate embeddings in batches
def generate_embeddings_batched(paper_generator, batch_size=BATCH_SIZE):
    batch = []
    for paper in tqdm(paper_generator, desc="Generating embeddings", unit="paper"):
        batch.append(paper)
        if len(batch) >= batch_size:
            yield from process_batch(batch)
            batch = []
    if batch:
        yield from process_batch(batch)

#Insert into DB in batches
def insert_embeddings_in_batches(data_gen, batch_size=2000):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO embeddings (paper_id, title_embedding, abstract_embedding, keywords_embedding)
        VALUES %s
        ON CONFLICT (paper_id) DO NOTHING;
    """
    batch = []
    count = 0
    for record in data_gen:
        batch.append(record)
        if len(batch) >= batch_size:
            execute_values(cursor, insert_query, batch)
            conn.commit()
            count += len(batch)
            tqdm.write(f"Inserted {count} embeddings...")
            batch = []
    if batch:
        execute_values(cursor, insert_query, batch)
        conn.commit()
        count += len(batch)
        tqdm.write(f"Inserted final batch. Total inserted: {count}")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    print("Loading paper metadata from DB...")
    db_papers = load_paper_data_from_db()

    print("Reading abstracts from JSON...")
    matched_papers = merge_with_abstracts(JSON_PATH, db_papers)

    print("Generating embeddings in batches...")
    embedding_gen = generate_embeddings_batched(matched_papers)

    print("Inserting into PostgreSQL...")
    insert_embeddings_in_batches(embedding_gen, batch_size=2000)
