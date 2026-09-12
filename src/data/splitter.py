import json
import random
import os
from typing import Dict, List, Any

def split_conversations(
    input_jsonl_path: str = "data/processed/amazon_conversations.jsonl",
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Splits data strictly at the CONVERSATION level to prevent data leakage.
    Never allows messages from the same conversation to appear across partitions.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading conversations from {input_jsonl_path}...")
    conversations: List[Dict[str, Any]] = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line.strip()))
                
    total_convs = len(conversations)
    print(f"Total conversations loaded: {total_convs:,}")
    
    # Shuffle conversations deterministically
    random.shuffle(conversations)
    
    n_train = int(total_convs * train_ratio)
    n_val = int(total_convs * val_ratio)
    
    train_set = conversations[:n_train]
    val_set = conversations[n_train:n_train + n_val]
    test_set = conversations[n_train + n_val:]
    
    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
    
    print("\nConversation-level split results:")
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
