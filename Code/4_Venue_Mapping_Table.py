import psycopg2
import csv

conn = psycopg2.connect(
    dbname='AMINER_V14',
    user='postgres',
    password='1234',
    host='localhost',
    port='5433'
)
cur = conn.cursor()

cur.execute("DELETE FROM venues_norm;")

# Insert mapping
with open("C:/Users/effik/Desktop/THESIS/test_postgre/Data/venue_mapping.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = row['raw_venue'].strip()
        norm = row['normalized_venue'].strip()
        if raw and norm:
            cur.execute("""
                INSERT INTO venues_norm (raw_venue, normalized_venue)
                VALUES (%s, %s)
                ON CONFLICT (raw_venue) DO UPDATE
                SET normalized_venue = EXCLUDED.normalized_venue;
            """, (raw, norm))

conn.commit()
cur.close()
conn.close()
print("venue_mapping.csv loaded into venues_norm.")
