import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# --- Config ---
EVAL_FOLDER = "C:/Users/effik/Desktop/THESIS/test_postgre/Data"
FAISS_INDEX_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/faiss_index.index"
METADATA_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/metadata.json"
CANONICAL_EMB_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/canonical_venue_embeddings.json"
EXCEL_OUTPUT = "C:/Users/effik/Desktop/THESIS/test_postgre/Results/Eval_Res_Rerank_Methods3.xlsx"
MODEL_NAME = "thenlper/gte-small"
TOP_K = [3, 5, 7]
SIM_THRESHOLD = 0.92

# --- Load models and data ---
embedder = SentenceTransformer(MODEL_NAME)
llm = OllamaLLM(model="mistral", temperature=0.0)

print("Loading FAISS index and metadata...")
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

# --- Citation-based reranker ---
def rerank_by_citation(candidate_entries, top_k=7):
    return sorted(candidate_entries, key=lambda r: r.get("n_citation", 0), reverse=True)[:top_k]

# --- Venue matcher ---
def match_venue_to_predictions(venue_name, predictions, threshold):
    venue_vec = embedder.encode(venue_name, normalize_embeddings=True).reshape(1, -1)
    sims = []
    for pred in predictions:
        pred_vec = embedder.encode(pred, normalize_embeddings=True).reshape(1, -1)
        sim = float(np.dot(venue_vec, pred_vec.T).item())
        sims.append(sim)
    return sims

# --- Evaluate across sets 1–5 ---
for set_idx in range(1, 6):
    print(f"\n=== Starting Evaluation Set {set_idx} (Citation-Based Reranking) ===")
    eval_path = os.path.join(EVAL_FOLDER, f"evaluation_set{set_idx}.json")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    hit_counts = {k: 0 for k in TOP_K}
    total = 0
    reciprocal_ranks = []
    ndcgs = []
    LOG_DISCOUNTS = np.log2(np.arange(2, 9))

    for idx, entry in enumerate(tqdm(eval_set, desc=f"Set {set_idx}")):
        abstract = entry["abstract"]
        raw_venue = entry["actual_venue"]

        query_emb = embedder.encode(abstract, normalize_embeddings=True).reshape(1, -1)
        _, I = index.search(query_emb, 20)
        candidates = [metadata[i] for i in I[0]]
        top_7 = rerank_by_citation(candidates)

        context = "\n".join(
            f'- "{r["title"]}" published in {r["venue"] or "Unknown"}' for r in top_7
        )

        try:
            response = rag_chain.invoke({"abstract": abstract, "similar_papers": context})
            lines = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
            preds = [line.split("(")[0].strip() if "(" in line else line for line in lines][:max(TOP_K)]
        except Exception:
            continue

        sims = match_venue_to_predictions(raw_venue, preds, SIM_THRESHOLD)
        paper_hits = {k: False for k in TOP_K}
        rr = 0

        for i, (pred, sim) in enumerate(zip(preds, sims)):
            for k in TOP_K:
                if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                    hit_counts[k] += 1
                    paper_hits[k] = True
            if sim >= SIM_THRESHOLD and rr == 0:
                rr = 1 / (i + 1)

        if not any(paper_hits.values()):
            try:
                expanded = llm.invoke(expansion_prompt.invoke({"venue": raw_venue})).strip()
                sims = match_venue_to_predictions(expanded, preds, SIM_THRESHOLD)
                for i, (pred, sim) in enumerate(zip(preds, sims)):
                    for k in TOP_K:
                        if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                            hit_counts[k] += 1
                            paper_hits[k] = True
                    if sim >= SIM_THRESHOLD and rr == 0:
                        rr = 1 / (i + 1)
            except Exception:
                pass

        reciprocal_ranks.append(rr)

        sims = np.array(sims[:7])
        if len(sims) < 7:
            sims = np.pad(sims, (0, 7 - len(sims)), constant_values=0)
        dcg = np.sum(sims / LOG_DISCOUNTS)
        idcg = np.sum(np.sort(sims)[::-1] / LOG_DISCOUNTS)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

        total += 1

    # --- Save results to Excel ---
    summary_df = pd.DataFrame({
        "Top-K": TOP_K,
        "Hits": [hit_counts[k] for k in TOP_K],
        "Total": total,
        "Hit Rate (%)": [round((hit_counts[k] / total) * 100, 2) for k in TOP_K]
    })

    extra_metrics_df = pd.DataFrame({
        "Metric": ["MRR", "Mean nDCG@7"],
        "Score": [round(np.mean(reciprocal_ranks), 4), round(np.mean(ndcgs), 4)]
    })

    with pd.ExcelWriter(EXCEL_OUTPUT, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name=f"Rerank Citations Set {set_idx}", index=False)
        extra_metrics_df.to_excel(writer, sheet_name=f"Rerank Citations Set {set_idx} - Extra", index=False)

    print(f"Finished Evaluation Set {set_idx}")
