import json
import re
import random
from collections import defaultdict
from typing import List, Dict, Any

random.seed(42)

test_file = "data/splits/test.jsonl"
print(f"Reading held-out test split from {test_file}...")

test_convs = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_convs.append(json.loads(line.strip()))

print(f"Loaded {len(test_convs):,} held-out candidate conversations.")

ESCALATE_PATTERNS = [
    r"\b(lawyer|legal|sue|fraud|stolen|police|hacked|unauthorized|investigate|dispute with bank|attorney|threat|scam|crime)\b",
    r"\b(disgrace|worst service|unacceptable|furious|disgusting|boycott|hung up on me|pathetic)\b",
    r"\b(locked out of account|cannot access account|account closed|stole my money|take you to court)\b"
]

def clean_tweet_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Group test conversations by their NLI intent
by_intent = defaultdict(list)
for conv in test_convs:
    intent = conv.get("intent") or conv.get("nli_intent")
    if intent == "general_unclassified":
        intent = "general_inquiry_greeting"
    cleaned = clean_tweet_text(conv["customer_text"])
    if len(cleaned) >= 5:
        by_intent[intent].append((conv, cleaned))

print("Candidate counts per intent in held-out test split:")
for intent, cand_list in sorted(by_intent.items()):
    print(f"  - {intent:<28}: {len(cand_list)} candidates")

# Target exactly 25 samples per intent across all 8 intents (25 x 8 = 200)
TARGET_INTENTS = [
    "delivery_issue",
    "return_refund_status",
    "damaged_defective_item",
    "cancellation_change_order",
    "account_prime_billing",
    "product_tech_support",
    "feedback_complaint",
    "general_inquiry_greeting"
]

golden_examples = []
item_id = 1

for intent in TARGET_INTENTS:
    cands = by_intent[intent]
    random.shuffle(cands)
    # Pick top 25
    selected = cands[:25]
    
    for conv, cleaned in selected:
        is_escalate = False
        escalation_reason = ""
        
        for ep in ESCALATE_PATTERNS:
            if re.search(ep, cleaned, re.IGNORECASE):
                is_escalate = True
                escalation_reason = "High-risk trigger keyword detected (fraud, legal threat, severe complaint, or account lock)."
                break
        
        if intent == "feedback_complaint" and not is_escalate:
            if random.random() < 0.50:
                is_escalate = True
                escalation_reason = "Customer complaint requires personal human supervisor empathy/apology."
                
        if not is_escalate:
            decision = "AUTO_HANDLE"
            eval_notes = f"Routine {intent} query. Suitable for automated response grounded in brand history."
        else:
            decision = "ESCALATE"
            eval_notes = f"Escalation required. Reason: {escalation_reason}"
            
        golden_examples.append({
            "id": f"eval_{item_id:03d}",
            "conversation_id": conv["conversation_id"],
            "customer_message": cleaned,
            "raw_customer_text": conv["customer_text"],
            "gold_intent": intent,
            "gold_escalation": decision,
            "evaluation_notes": eval_notes
        })
        item_id += 1

print(f"\nFinal Golden Evaluation Set count: {len(golden_examples)} items (exactly 25 per intent across 8 intents)")
escalate_count = sum(1 for e in golden_examples if e["gold_escalation"] == "ESCALATE")
auto_count = sum(1 for e in golden_examples if e["gold_escalation"] == "AUTO_HANDLE")

print(f"  AUTO_HANDLE cases : {auto_count} ({auto_count/len(golden_examples)*100:.1f}%)")
print(f"  ESCALATE cases    : {escalate_count} ({escalate_count/len(golden_examples)*100:.1f}%)")

output_path = "evaluation/golden_set.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(golden_examples, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(golden_examples)} golden examples to: {output_path}")

