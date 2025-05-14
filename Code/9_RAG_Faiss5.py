import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# --- Config ---
EVAL_FILE = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/evaluation_set2.json"
FAISS_INDEX_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/faiss_index.index"
METADATA_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/metadata.json"
CANONICAL_EMB_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/canonical_venue_embeddings.json"
MODEL_NAME = "thenlper/gte-small"
TOP_K = [3, 5, 7]
SIM_THRESHOLD = 0.92

# --- Load models and data ---
embedder = SentenceTransformer(MODEL_NAME)
llm = OllamaLLM(model="mistral", temperature=0.0)

print("Loading FAISS, metadata, and canonical embeddings...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
with open(CANONICAL_EMB_PATH, "r", encoding="utf-8") as f:
    canonical_data = json.load(f)

canonical_names = [x["name"] for x in canonical_data]
canonical_embeddings = np.array([x["embedding"] for x in canonical_data], dtype=np.float32)
canonical_embeddings = canonical_embeddings / np.linalg.norm(canonical_embeddings, axis=1, keepdims=True)

# --- Prompts ---
expansion_prompt = PromptTemplate.from_template("""
You are a smart assistant that converts abbreviated or informal venue names into their standardized full academic names.

Only respond with the formal name used in academic citations, without any commentary or extra information.

If the name is already complete, return it exactly as is.

Venue name:
"{venue}"
""")

rag_prompt = PromptTemplate.from_template("""
You are an expert academic assistant. Based on the abstract of a new paper and a list of similar papers, recommend the most appropriate publication venues.

Only consider venues where similar papers were actually published. Use the provided venues directly without considering citation counts.

Respond with a ranked list of 7 academic venue names. Use the full official name of each venue. Do not explain or comment on your choices.

New paper abstract:
{abstract}

Similar papers (with venues):
{similar_papers}

Format your response as a simple list:
- Venue Name 1
- Venue Name 2
- Venue Name 3
...
""")

rag_chain = rag_prompt | llm

# --- Load evaluation set ---
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_set = json.load(f)

hit_counts = {k: 0 for k in TOP_K}
total = 0

print("🚀 Starting Evaluation...\n")

# --- Load all embeddings from FAISS (assumed same order as metadata) ---
all_embeddings = index.reconstruct_n(0, index.ntotal)

# --- Embedding-based reranker ---
def rerank_candidates_by_embedding(query_emb, indices):
    scored = [(float(np.dot(query_emb, all_embeddings[i])), i, metadata[i]) for i in indices]
    ranked = [r for _, _, r in sorted(scored, reverse=True)]
    return ranked[:7]

def match_venue_to_predictions(venue_name, predictions, threshold):
    venue_vec = embedder.encode(venue_name, normalize_embeddings=True).reshape(1, -1)
    sims = []
    for pred in predictions:
        pred_vec = embedder.encode(pred, normalize_embeddings=True).reshape(1, -1)
        sim = float(np.dot(venue_vec, pred_vec.T).item())
        sims.append(sim)
    return sims

for idx, entry in enumerate(eval_set):
    print(f"\n📄 Paper {idx + 1}")
    abstract = entry["abstract"]
    raw_venue = entry["actual_venue"]
    print(f"📌 Actual venue: {raw_venue}")

    # Step 1: FAISS + rerank
    query_emb = embedder.encode(abstract, normalize_embeddings=True).reshape(1, -1)
    _, I = index.search(query_emb, 20)  # fetch more candidates before reranking
    top_7 = rerank_candidates_by_embedding(query_emb.flatten(), I[0])

    context = "\n".join(
        f'- "{r["title"]}" published in {r["venue"] or "Unknown"}' for r in top_7
    )

    # Step 2: LLM prediction
    try:
        response = rag_chain.invoke({"abstract": abstract, "similar_papers": context})
        lines = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
        preds = [line.split("(")[0].strip() if "(" in line else line for line in lines][:max(TOP_K)]
    except Exception as e:
        print(f"⚠️ LLM failed on idx={idx}: {e}")
        continue

    # Step 3: Try raw venue match
    sims = match_venue_to_predictions(raw_venue, preds, SIM_THRESHOLD)
    paper_hits = {k: False for k in TOP_K}
    for i, (pred, sim) in enumerate(zip(preds, sims)):
        print(f"🔍 Prediction {i + 1}: {pred} → sim = {sim:.4f}")
        for k in TOP_K:
            if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                hit_counts[k] += 1
                paper_hits[k] = True

    if not any(paper_hits.values()):
        try:
            expanded = llm.invoke(expansion_prompt.invoke({"venue": raw_venue})).strip()
            print(f"🧠 Used LLM-expanded venue: {raw_venue} → {expanded}")
            sims = match_venue_to_predictions(expanded, preds, SIM_THRESHOLD)
            for i, (pred, sim) in enumerate(zip(preds, sims)):
                print(f"🔍 Expanded match {i + 1}: {pred} → sim = {sim:.4f}")
                for k in TOP_K:
                    if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                        hit_counts[k] += 1
                        paper_hits[k] = True
        except Exception as e:
            print(f"⚠️ LLM expansion failed: {e}")

    for k in TOP_K:
        print(f"Top-{k} Hit: {'✅' if paper_hits[k] else '❌'}")

    total += 1

# --- Final Report ---
print("\n📈 Final Evaluation Results (With Conditional Expansion):")
for k in TOP_K:
    print(f"Top-{k} Hit Rate: {hit_counts[k]} / {total} = {(hit_counts[k] / total) * 100:.2f}%")