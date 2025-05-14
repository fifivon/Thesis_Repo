import csv
import re
import html
from rapidfuzz import fuzz
from tqdm import tqdm

# === CONFIG ===
INPUT_CSV = r"C:/Users/effik/Desktop/THESIS/test_postgre/Data/venues.csv"                # Input: raw venue list from DB
MAPPING_CSV = r"C:/Users/effik/Desktop/THESIS/test_postgre/Data/venue_mapping.csv"       # Output: raw → normalized
SIMILARITY_THRESHOLD = 93               # Fuzzy match threshold

# === Bad Venue Terms ===
BAD_VENUE_TERMS = {
    "", "-", "unknown", "null", "none", "test", "...", "n/a", "na", "unpublished", "zzz"
}

# === Normalization Function (no acronym fixes) ===
def normalize_venue_name(name: str) -> str:
    if not name:
        return ""

    name = html.unescape(name)                         # Decode HTML entities
    name = name.encode("ascii", "ignore").decode()     # Remove accents
    name = name.lower().strip()

    # Remove year patterns like 2022 or 1998
    name = re.sub(r"\b(19|20)\d{2}\b", "", name)

    # Remove volume/issue/part info
    name = re.sub(r"\b\d+\s*(st|nd|rd|th)?\b", "", name)
    name = re.sub(r"\b(volume|vol\.?|part|issue|edition|no\.?)\s*(\d+|[ivxlcdm]+)?\b", "", name)

    # Remove Roman numerals
    name = re.sub(r"\b(?=[mdclxvi]+\b)m{0,4}(cm|cd|d?c{0,3})?(xc|xl|l?x{0,3})?(ix|iv|v?i{0,3})?\b", "", name)

    # Remove acronyms or short codes in parentheses
    name = re.sub(r"\([^)]{2,15}\)", "", name)

    # Clean up special characters
    name = name.replace("&", "and")
    name = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", name)
    name = re.sub(r"[\"'“”‘’.]", "", name)
    name = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", name)
    name = re.sub(r"\s+", " ", name)

    # Final format
    name = name.strip().title()

    # Filter out junk
    if name.lower() in BAD_VENUE_TERMS or len(name) < 3:
        return ""

    if re.fullmatch(r"[a-z]{1,3}", name.lower()):
        return ""

    return name

# === MAIN LOGIC ===
seen_normalized = []
mapping = []
skipped = 0

with open(INPUT_CSV, "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)
    # Deduplicate + strip
    raw_venues = list(set(row["venue"].strip() for row in reader))

for raw_venue in tqdm(raw_venues, desc="Fuzzy matching"):
    norm = normalize_venue_name(raw_venue)
    if not norm:
        skipped += 1
        continue

    matched = False
    for existing in seen_normalized:
        if fuzz.ratio(norm, existing) >= SIMILARITY_THRESHOLD:
            mapping.append((raw_venue, existing))
            matched = True
            break

    if not matched:
        seen_normalized.append(norm)
        mapping.append((raw_venue, norm))

# === SAVE TO CSV ===
with open(MAPPING_CSV, "w", newline='', encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["raw_venue", "normalized_venue"])
    for row in mapping:
        writer.writerow(row)

print(f"\nDone! {len(mapping)} raw venues processed and saved to '{MAPPING_CSV}'")
print(f"Skipped {skipped} venues due to empty or junk normalization.")
