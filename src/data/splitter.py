import json
import random
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any

def split_conversations(
    input_jsonl_path: str = "data/processed/amazon_conversations_nli_tagged.jsonl",
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Stratified Conversation-Level Splitter.
    Ensures identical 8-intent class distribution across Train (70%), Val (15%), and Test (15%)
    while guaranteeing 0% conversation ID overlap (zero data leakage).
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading conversations from {input_jsonl_path}...")
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_convs = 0
    
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                # Standardize intent tag
                intent = item.get("nli_intent", "general_unclassified")
                if intent == "general_unclassified":
                    intent = "general_inquiry_greeting"
                item["intent"] = intent
                by_intent[intent].append(item)
                total_convs += 1
                
    print(f"Total conversations loaded: {total_convs:,} across {len(by_intent)} intents.")
    
    train_set = []
    val_set = []
    test_set = []
    
    for intent, items in by_intent.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])
        
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
    
    print("\nStratified Conversation-Level Split Results:")
    for split_name, data in splits.items():
        out_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        pct = (len(data) / total_convs) * 100
        print(f"  - {split_name.upper():<5}: {len(data):,} conversations ({pct:.1f}%) -> {out_file}")

    print("\nLeakage verification: Verifying zero overlap in conversation IDs...")
    train_ids = set(c["conversation_id"] for c in train_set)
    val_ids = set(c["conversation_id"] for c in val_set)
    test_ids = set(c["conversation_id"] for c in test_set)
    
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)
    
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("VERIFIED: 0 conversation overlap! Strict isolation confirmed.")
    else:
        raise ValueError("CRITICAL ERROR: Data leakage detected across splits!")

if __name__ == "__main__":
    split_conversations()

