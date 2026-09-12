import json
import numpy as np
from typing import Dict, Any, List
from src.retrieval.retriever import HistoricalCaseRetriever

def evaluate_retrieval_system(
    golden_set_path: str = "evaluation/golden_set.json",
    top_k_eval: int = 5
) -> Dict[str, Any]:
    print("=" * 65)
    print("PHASE 9: RETRIEVAL EVALUATION (NAIVE VS INTENT-AWARE)")
    print("=" * 65)
    
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    retriever = HistoricalCaseRetriever()
    
    naive_recalls_at_1 = []
    naive_recalls_at_3 = []
    naive_recalls_at_5 = []
    naive_mrrs = []
    
    intent_recalls_at_1 = []
    intent_recalls_at_3 = []
    intent_recalls_at_5 = []
    intent_mrrs = []
    
    print(f"\nEvaluating on all {len(golden_data)} Golden Set queries...")
    
    for item in golden_data:
        query = item["customer_message"]
        gold_intent = item["gold_intent"]
        
        # 1. Evaluate Naive Retrieval
        naive_cases = retriever.retrieve_naive(query, top_k=top_k_eval)
        naive_intents = [c["intent"] for c in naive_cases]
        
        # Recall@K: is gold_intent present in top K?
        r1_naive = 1.0 if len(naive_intents) > 0 and naive_intents[0] == gold_intent else 0.0
        r3_naive = 1.0 if gold_intent in naive_intents[:3] else 0.0
        r5_naive = 1.0 if gold_intent in naive_intents[:5] else 0.0
        
        # MRR: 1 / rank of first appearance of gold_intent
        mrr_naive = 0.0
        for rank, intent in enumerate(naive_intents, 1):
            if intent == gold_intent:
                mrr_naive = 1.0 / rank
                break
                
        naive_recalls_at_1.append(r1_naive)
        naive_recalls_at_3.append(r3_naive)
        naive_recalls_at_5.append(r5_naive)
        naive_mrrs.append(mrr_naive)
        
        # 2. Evaluate Intent-Aware Retrieval
        intent_cases = retriever.retrieve_intent_aware(query, intent=gold_intent, top_k=top_k_eval)
        intent_intents = [c["intent"] for c in intent_cases]
        
        r1_intent = 1.0 if len(intent_intents) > 0 and intent_intents[0] == gold_intent else 0.0
        r3_intent = 1.0 if gold_intent in intent_intents[:3] else 0.0
        r5_intent = 1.0 if gold_intent in intent_intents[:5] else 0.0
        
        mrr_intent = 0.0
        for rank, intent in enumerate(intent_intents, 1):
            if intent == gold_intent:
                mrr_intent = 1.0 / rank
                break
                
        intent_recalls_at_1.append(r1_intent)
        intent_recalls_at_3.append(r3_intent)
        intent_recalls_at_5.append(r5_intent)
        intent_mrrs.append(mrr_intent)
        
    results = {
        "naive": {
            "Recall@1": float(round(np.mean(naive_recalls_at_1), 4)),
            "Recall@3": float(round(np.mean(naive_recalls_at_3), 4)),
            "Recall@5": float(round(np.mean(naive_recalls_at_5), 4)),
            "MRR": float(round(np.mean(naive_mrrs), 4))
        },
        "intent_aware": {
            "Recall@1": float(round(np.mean(intent_recalls_at_1), 4)),
            "Recall@3": float(round(np.mean(intent_recalls_at_3), 4)),
            "Recall@5": float(round(np.mean(intent_recalls_at_5), 4)),
            "MRR": float(round(np.mean(intent_mrrs), 4))
        }
    }
    
    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION RESULTS (ON 200 GOLDEN SET EXAMPLES):")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Naive Semantic Search':<25} | {'Intent-Aware Retrieval':<25}")
    print("-" * 70)
    for m in ["Recall@1", "Recall@3", "Recall@5", "MRR"]:
        n_val = results["naive"][m]
        i_val = results["intent_aware"][m]
        print(f"{m:<15} | {n_val:<25.4f} | {i_val:<25.4f}")
        
    return results

if __name__ == "__main__":
    evaluate_retrieval_system()
