import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Config ---
EVAL_FILE = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/evaluation_set2.json"
CANONICAL_EMB_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/canonical_venue_embeddings.json"
MODEL_NAME = "thenlper/gte-small"
TOP_K = [3, 5, 7]
SIM_THRESHOLD = 0.92

# --- Load models ---
embedder = SentenceTransformer(MODEL_NAME)
llm = OllamaLLM(model="mistral", temperature=0.0)

# --- Load canonical venue embeddings ---
with open(CANONICAL_EMB_PATH, encoding="utf-8") as f:
    canonical_data = json.load(f)
canonical_names = [x["name"] for x in canonical_data]
canonical_embeddings = np.array([x["embedding"] for x in canonical_data])
canonical_embeddings = canonical_embeddings / np.linalg.norm(canonical_embeddings, axis=1, keepdims=True)

# --- Load evaluation set ---
with open(EVAL_FILE, encoding="utf-8") as f:
    eval_data = json.load(f)

# --- Prompt for transforming actual venue ---
expansion_prompt = PromptTemplate.from_template("""
You are a smart assistant that expands abbreviated or informal academic venue names to their full official names used in academic citations.

Respond with the expanded full name only, no commentary.

Original venue name:
"{venue}"
""")

# --- Prompt for predicting venues (no RAG) ---
llm_only_prompt = PromptTemplate.from_template("""
You are an expert research assistant. Based only on the following paper abstract, suggest the top 7 academic venues where this paper could be submitted.

Abstract:
{abstract}

Respond with a simple list of full venue names only, one per line.
""")
llm_chain = llm_only_prompt | llm

# --- Evaluation ---
hit_counts = {k: 0 for k in TOP_K}
total = 0

print("\nStarting Evaluation (LLM-Only with LLM-Transformed Ground Truth)...\n")

for idx, entry in enumerate(tqdm(eval_data, desc="Evaluating")):
    abstract = entry["abstract"]
    actual_raw = entry["actual_venue"]

    # Step 1: Expand actual venue using LLM
    try:
        transformed = llm.invoke(expansion_prompt.invoke({"venue": actual_raw})).strip()
    except Exception as e:
        print(f"Expansion failed at idx={idx}: {e}")
        transformed = actual_raw

    actual_vec = embedder.encode(transformed, normalize_embeddings=True).reshape(1, -1)
    sims = (actual_vec @ canonical_embeddings.T).flatten()
    actual_index = int(np.argmax(sims))
    actual_canonical = canonical_names[actual_index]
    sim_to_canonical = float(sims[actual_index])

    if sim_to_canonical >= SIM_THRESHOLD:
        actual_emb = canonical_embeddings[actual_index].reshape(1, -1)
    else:
        actual_emb = actual_vec
        actual_canonical = transformed

    print(f"\nPaper {idx + 1}")
    print(f"Actual venue: {actual_raw}")
    print(f"LLM-transformed venue: {transformed} → matched to: {actual_canonical}")

    # Step 2: LLM-only prediction
    try:
        response = llm_chain.invoke({"abstract": abstract})
        preds = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
    except Exception as e:
        print(f"LLM failed on idx={idx}: {e}")
        continue

    paper_hits = {k: False for k in TOP_K}

    for i, pred in enumerate(preds[:max(TOP_K)]):
        pred_vec = embedder.encode(pred, normalize_embeddings=True).reshape(1, -1)
        pred_sims = (pred_vec @ canonical_embeddings.T).flatten()
        pred_index = int(np.argmax(pred_sims))
        pred_sim = float(np.dot(actual_emb, canonical_embeddings[pred_index]).item())

        print(f"Prediction {i + 1}: {pred} → sim = {pred_sim:.4f}")

        for k in TOP_K:
            if i < k and not paper_hits[k] and pred_sim >= SIM_THRESHOLD:
                hit_counts[k] += 1
                paper_hits[k] = True

    for k in TOP_K:
        print(f"Top-{k} Hit: {'H' if paper_hits[k] else 'NH'}")

    total += 1

# --- Final Report ---
print("\nFinal Evaluation Results:")
for k in TOP_K:
    print(f"Top-{k} Hit Rate: {hit_counts[k]} / {total} = {(hit_counts[k] / total) * 100:.2f}%")
