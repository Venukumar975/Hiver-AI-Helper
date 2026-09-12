import json
from collections import Counter
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, classification_report

def run_majority_baseline(
    golden_set_path: str = "evaluation/golden_set.json",
    train_split_path: str = "data/splits/train.jsonl"
) -> Dict[str, Any]:
    print("=" * 60)
    print("BASELINE 1: TRIVIAL MAJORITY CLASS CLASSIFIER")
    print("=" * 60)
    
    # Load golden evaluation set
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    y_true = [item["gold_intent"] for item in golden_data]
    
    # The most common intent in routine customer support
    intent_counts = Counter(y_true)
    majority_intent, majority_count = intent_counts.most_common(1)[0]
    print(f"Majority class identified: '{majority_intent}' ({majority_count}/{len(y_true)} samples)")
    
    # Predict majority class for all inputs
    y_pred = [majority_intent] * len(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print("\n--- RESULTS ON GOLDEN EVALUATION SET (200 SAMPLES) ---")
    print(f"Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    results = {
        "model": "MajorityClassBaseline",
        "majority_class": majority_intent,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }
    return results

if __name__ == "__main__":
    run_majority_baseline()
