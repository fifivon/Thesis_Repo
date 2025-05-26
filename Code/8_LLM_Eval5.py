import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from datetime import datetime


EVAL_FOLDER = "C:/Users/effik/Desktop/THESIS/test_postgre/Data"
CANONICAL_EMB_PATH = "C:/Users/effik/Desktop/THESIS/test_postgre/Data/canonical_venue_embeddings.json"
EXCEL_OUTPUT = "C:/Users/effik/Desktop/THESIS/test_postgre/Results/Res_LLM_only.xlsx"
MODEL_NAME = "thenlper/gte-small"
TOP_K = [3, 5, 7]
SIM_THRESHOLD = 0.92
TIMEOUT = 120  #seconds


embedder = SentenceTransformer(MODEL_NAME)
llm = OllamaLLM(model="mistral", temperature=0.0)

#Load canonical embeddings
with open(CANONICAL_EMB_PATH, encoding="utf-8") as f:
    canonical_data = json.load(f)
canonical_names = [x["name"] for x in canonical_data]
canonical_embeddings = np.array([x["embedding"] for x in canonical_data], dtype=np.float32)
canonical_embeddings = canonical_embeddings / np.linalg.norm(canonical_embeddings, axis=1, keepdims=True)

#Prompts
expansion_prompt = PromptTemplate.from_template("""
You are a smart assistant that converts abbreviated or informal venue names into their standardized full academic names.
Only respond with the formal name used in academic citations, without any commentary or extra information.
If the name is already complete, return it exactly as is.

Venue name:
"{venue}"
""")

llm_only_prompt = PromptTemplate.from_template("""
You are an expert research assistant. Based only on the following paper abstract, suggest the top 7 academic venues where this paper could be submitted.

Abstract:
{abstract}

Respond with a simple list of full venue names only, one per line.
""")

llm_chain = llm_only_prompt | llm
expansion_chain = expansion_prompt | llm

#Timeout-safe LLM call
def safe_llm_invoke(chain, input_dict, timeout=TIMEOUT, context="unknown"):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(chain.invoke, input_dict)
        try:
            print(f"[{datetime.now()}] Calling LLM for: {context}")
            result = future.result(timeout=timeout)
            print(f"[{datetime.now()}] LLM returned for: {context}")
            return result
        except concurrent.futures.TimeoutError:
            print(f"[{datetime.now()}] LLM call timed out on: {context}")
            return None
        except Exception as e:
            print(f"[{datetime.now()}] LLM exception on {context}: {e}")
            return None

#Loop over evaluation sets
for set_idx in range(8, 11):
    eval_file = os.path.join(EVAL_FOLDER, f"evaluation_set{set_idx}.json")
    print(f"\n=== Starting Evaluation Set {set_idx} ===")

    with open(eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)

    hit_counts = {k: 0 for k in TOP_K}
    total = 0
    timeout_count = 0
    reciprocal_ranks = []
    ndcgs = []
    LOG_DISCOUNTS = np.log2(np.arange(2, 9))

    for idx, entry in enumerate(tqdm(eval_data, desc=f"Set {set_idx}")):
        abstract = entry["abstract"]
        raw_venue = entry["actual_venue"]

        response = safe_llm_invoke(llm_chain, {"abstract": abstract}, context=f"abstract {idx + 1}")
        if response is None:
            timeout_count += 1
            continue

        preds = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
        if not preds:
            print(f"[{datetime.now()}] Empty prediction list for paper {idx + 1}. Skipping.")
            continue

        paper_hits = {k: False for k in TOP_K}
        sims = []
        rr = 0

        actual_vec = embedder.encode(raw_venue, normalize_embeddings=True).reshape(1, -1)
        for i, pred in enumerate(preds[:7]):
            pred_vec = embedder.encode(pred, normalize_embeddings=True).reshape(1, -1)
            sim = float(np.dot(actual_vec, pred_vec.T).item())
            sims.append(sim)
            for k in TOP_K:
                if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                    hit_counts[k] += 1
                    paper_hits[k] = True
            if sim >= SIM_THRESHOLD and rr == 0:
                rr = 1 / (i + 1)

        if not any(paper_hits.values()):
            transformed = safe_llm_invoke(expansion_chain, {"venue": raw_venue}, context=f"expansion {idx + 1}")
            if transformed is not None:
                transformed = transformed.strip()
                actual_vec = embedder.encode(transformed, normalize_embeddings=True).reshape(1, -1)
                for i, pred in enumerate(preds[:7]):
                    pred_vec = embedder.encode(pred, normalize_embeddings=True).reshape(1, -1)
                    sim = float(np.dot(actual_vec, pred_vec.T).item())
                    sims[i] = sim
                    for k in TOP_K:
                        if i < k and not paper_hits[k] and sim >= SIM_THRESHOLD:
                            hit_counts[k] += 1
                            paper_hits[k] = True
                    if sim >= SIM_THRESHOLD and rr == 0:
                        rr = 1 / (i + 1)

        sims = np.array(sims[:7])
        if len(sims) < 7:
            sims = np.pad(sims, (0, 7 - len(sims)), constant_values=0)
        dcg = np.sum(sims / LOG_DISCOUNTS)
        idcg = np.sum(np.sort(sims)[::-1] / LOG_DISCOUNTS)
        ndcg = dcg / idcg if idcg > 0 else 0

        reciprocal_ranks.append(rr)
        ndcgs.append(ndcg)
        total += 1

    # Saving Results
    summary_df = pd.DataFrame({
        "Top-K": TOP_K,
        "Hits": [hit_counts[k] for k in TOP_K],
        "Total": total,
        "Hit Rate (%)": [round((hit_counts[k] / total) * 100, 2) for k in TOP_K]
    })

    extra_metrics_df = pd.DataFrame({
        "Metric": ["MRR", "Mean nDCG@7", "Timeouts"],
        "Score": [round(np.mean(reciprocal_ranks), 4), round(np.mean(ndcgs), 4), timeout_count]
    })

    with pd.ExcelWriter(EXCEL_OUTPUT, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name=f"LLM Only Set {set_idx}", index=False)
        extra_metrics_df.to_excel(writer, sheet_name=f"LLM Only Set {set_idx} - Extra", index=False)

    print(f"Finished Evaluation Set {set_idx}: Results saved.")
    print(f"Total timeouts in Set {set_idx}: {timeout_count}")
