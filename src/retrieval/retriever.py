import os
import re
import faiss
import joblib
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class HistoricalCaseRetriever:
    """
    Industry-standard Vector Retriever powered by FAISS and Sentence-Transformers.
    Supports:
      1. Naive Semantic Search (search across all 15,000 cases)
      2. Intent-Aware Filtered Search (filter by predicted intent, then rank by cosine similarity)
    """
    def __init__(self, artifacts_dir: str = "artifacts/retrieval"):
        self.index_path = os.path.join(artifacts_dir, "historical_cases.faiss")
        self.metadata_path = os.path.join(artifacts_dir, "cases_metadata.joblib")
        
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(self.index_path)
        self.cases = joblib.load(self.metadata_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Pre-build intent-to-indices mapping for fast Intent-Aware filtering
        self.intent_indices = {}
        for idx, case in enumerate(self.cases):
            intent = case.get("intent", "general")
            if intent not in self.intent_indices:
                self.intent_indices[intent] = []
            self.intent_indices[intent].append(idx)
            
        print(f"Retriever initialized! {self.index.ntotal} historical cases indexed across {len(self.intent_indices)} intent buckets.")

    def retrieve_naive(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Standard Semantic Search across entire vector database."""
        cleaned = clean_text(query)
        q_vec = self.embedder.encode([cleaned], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")
        
        scores, indices = self.index.search(q_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            case = self.cases[idx]
            results.append({
                "case_id": case["case_id"],
                "similarity_score": float(round(score, 4)),
                "intent": case["intent"],
                "customer_issue": case["customer_issue"],
                "brand_resolution": case["brand_resolution"]
            })
        return results

    def retrieve_intent_aware(self, query: str, intent: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Intent-Aware Filtered Semantic Search.
        First restricts the search space to cases matching the predicted intent,
        then finds the top-K semantically closest resolutions.
        """
        if not intent or intent not in self.intent_indices or len(self.intent_indices[intent]) == 0:
            # Fallback to naive if unknown intent
            return self.retrieve_naive(query, top_k)
            
        candidate_indices = np.array(self.intent_indices[intent], dtype=np.int64)
        
        cleaned = clean_text(query)
        q_vec = self.embedder.encode([cleaned], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")[0]
        
        # Reconstruct candidate embeddings from FAISS index to compute cosine similarity
        cand_vecs = np.zeros((len(candidate_indices), self.index.d), dtype="float32")
        for i, idx in enumerate(candidate_indices):
            cand_vecs[i] = self.index.reconstruct(int(idx))
            
        # Inner product (cosine similarity on normalized vectors)
        sim_scores = np.dot(cand_vecs, q_vec)
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            orig_idx = candidate_indices[i]
            case = self.cases[orig_idx]
            results.append({
                "case_id": case["case_id"],
                "similarity_score": float(round(sim_scores[i], 4)),
                "intent": case["intent"],
                "customer_issue": case["customer_issue"],
                "brand_resolution": case["brand_resolution"]
            })
        return results

if __name__ == "__main__":
    retriever = HistoricalCaseRetriever()
    test_query = "My package has not arrived and tracking says delivered!"
    
    print("\n" + "=" * 60)
    print(f"QUERY: '{test_query}'")
    print("=" * 60)
    
    print("\n--- 1. NAIVE RETRIEVAL (TOP 3) ---")
    naive_res = retriever.retrieve_naive(test_query, top_k=3)
    for r in naive_res:
        print(f"[{r['case_id']}] Score: {r['similarity_score']} | Intent: {r['intent']}")
        print(f"  Issue: {r['customer_issue']}")
        print(f"  Resolution: {r['brand_resolution']}\n")
        
    print("\n--- 2. INTENT-AWARE RETRIEVAL (Intent: delivery_issue) ---")
    intent_res = retriever.retrieve_intent_aware(test_query, intent="delivery_issue", top_k=3)
    for r in intent_res:
        print(f"[{r['case_id']}] Score: {r['similarity_score']} | Intent: {r['intent']}")
        print(f"  Issue: {r['customer_issue']}")
        print(f"  Resolution: {r['brand_resolution']}\n")
