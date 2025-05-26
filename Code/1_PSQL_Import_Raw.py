import json
import psycopg2
from psycopg2.extras import execute_values
import ijson

json_path = r"C:/Users/effik/Downloads/dblp_v14/dblp_v14.json"
db_config = {
    'dbname': 'AMINER_V14',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433'
}

def load_and_clean_data(path, batch_size=10000):
    with open(path, 'r', encoding='utf-8') as f:
        papers = ijson.items(f, 'item')
        batch = []
        for paper in papers:
            try:
                # Skip if there's no venue or venue.raw
                if not paper.get('venue') or not paper['venue'].get('raw'):
                    continue

                # Skip if abstract is missing or empty
                if not paper.get('abstract'):
                    continue

                # Skip papers with 0 citations
                if paper.get('n_citation', 0) == 0:
                    continue

                # Skip non-latin characters papers
                if paper.get('lang') == 'zh':
                    continue

                batch.append({
                    'id': paper.get('id'),
                    'title': paper.get('title'),
                    'author_names': [a.get('name') for a in paper.get('authors', []) if a.get('name')],
                    'author_orgs': [a.get('org') for a in paper.get('authors', []) if a.get('org')],
                    'author_ids': [a.get('id') for a in paper.get('authors', []) if a.get('id')],
                    'venue': paper['venue'].get('raw'),
                    'year': paper.get('year'),
                    'keywords': paper.get('keywords', []),
                    'fos_names': [f.get('name') for f in paper.get('fos', []) if f.get('name')],
                    'fos_weights': [f.get('w') for f in paper.get('fos', []) if f.get('w')],
                    'paper_references': paper.get('references', []),
                    'n_citation': paper.get('n_citation'),
                    'page_start': paper.get('page_start'),
                    'page_end': paper.get('page_end'),
                    'doc_type': paper.get('doc_type'),
                    'lang': paper.get('lang'),
                    'volume': paper.get('volume'),
                    'issue': paper.get('issue'),
                    'issn': paper.get('issn'),
                    'isbn': paper.get('isbn'),
                    'doi': paper.get('doi'),
                    'url': paper.get('url', [])
                })

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                print(f"Skipping record due to error: {e}")
        if batch:
            yield batch

def insert_to_postgres(data, config):
    conn = psycopg2.connect(**config)
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO papers (
        id, title, author_names, author_orgs, author_ids,
        venue, year, keywords, fos_names, fos_weights, paper_references,
        n_citation, page_start, page_end, doc_type, lang,
        volume, issue, issn, isbn, doi, url
    )
    VALUES %s
    ON CONFLICT (id) DO NOTHING;
    """

    values = [
        (
            d['id'], d['title'], d['author_names'], d['author_orgs'], d['author_ids'],
            d['venue'], d['year'], d['keywords'], d['fos_names'], d['fos_weights'], d['paper_references'],
            d['n_citation'], d['page_start'], d['page_end'], d['doc_type'], d['lang'],
            d['volume'], d['issue'], d['issn'], d['isbn'], d['doi'], d['url']
        )
        for d in data
    ]

    execute_values(cursor, insert_query, values)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Inserted {len(values)} records.")

if __name__ == "__main__":
    total_inserted = 0
    print("Cleaning and inserting in batches...")
    for i, batch in enumerate(load_and_clean_data(json_path, batch_size=10000), start=1):
        print(f"Inserting batch {i} with {len(batch)} records...")
        insert_to_postgres(batch, db_config)
        total_inserted += len(batch)

    print(f"Total records inserted: {total_inserted}")
