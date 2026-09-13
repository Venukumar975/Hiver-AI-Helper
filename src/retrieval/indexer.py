import json
import os
import re
import joblib
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_resolution_index(
    train_split_path: str = "data/splits/train.jsonl",
    output_dir: str = "artifacts/retrieval",
    max_cases: int = 15000  # Index 15,000 high quality cases for fast search & light storage
):
    print("=" * 65)
    print("BUILDING HISTORICAL RESOLUTION FAISS VECTOR DATABASE")
    print("=" * 65)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cases: List[Dict[str, Any]] = []
    texts_to_embed: List[str] = []
    
    print(f"Loading training cases from {train_split_path}...")
    with open(train_split_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(cases) >= max_cases:
                break
            data = json.loads(line.strip())
            c_text = clean_text(data["customer_text"])
            b_text = clean_text(data["brand_text"])
            
            if len(c_text) < 15 or len(b_text) < 15:
                continue
                
            assigned_intent = data.get("intent") or data.get("nli_intent", "general_inquiry_greeting")
            if assigned_intent == "general_unclassified":
                assigned_intent = "general_inquiry_greeting"
                    
            cases.append({
                "case_id": data["conversation_id"],
                "customer_issue": c_text,
                "brand_resolution": b_text,
                "intent": assigned_intent
            })
            texts_to_embed.append(c_text)

    print(f"Loaded {len(cases):,} historical resolved cases.")
    
    # Generate embeddings
    print("Loading SentenceTransformer('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding embeddings for historical cases...")
    embeddings = embedder.encode(texts_to_embed, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")
    
    # Build FAISS Index (IndexFlatIP for cosine similarity on normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built! Total vectors indexed: {index.ntotal}")
    
    # Save artifacts
    index_path = os.path.join(output_dir, "historical_cases.faiss")
    cases_path = os.path.join(output_dir, "cases_metadata.joblib")
    
    faiss.write_index(index, index_path)
    joblib.dump(cases, cases_path)
    
    print(f"Saved FAISS index to: {index_path}")
    print(f"Saved Metadata to   : {cases_path}")

if __name__ == "__main__":
    build_resolution_index()
