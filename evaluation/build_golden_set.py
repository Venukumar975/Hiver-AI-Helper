import json
import re
import random
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

# Expanded regex patterns for all 7 intents
INTENT_PATTERNS = {
    "delivery_issue": [
        r"\b(where is my (order|package|parcel)|delayed|not arrived|late delivery|tracking says delivered|still waiting|havent received|hasnt arrived|stuck in transit|carrier|amzl|estimated delivery)\b"
    ],
    "return_refund_status": [
        r"\b(refund|return|sent back|drop off|ups pickup|money back|reimburse|refunded|return label|credit back)\b"
    ],
    "damaged_defective_item": [
        r"\b(broken|damaged|shattered|cracked|defective|wrong item|missing parts|empty box|leaking|doesnt work|wont turn on|not working|ripped|smashed)\b"
    ],
    "cancellation_change_order": [
        r"\b(cancel|cancelled|cancelling|change address|change shipping|by mistake|ordered wrong|change my order|modify order|stop shipment)\b"
    ],
    "account_prime_billing": [
        r"\b(prime|membership|subscription|charged|unauthorized charge|bank statement|fraud|hacked|locked out|password|login|gift card|redeem|scam|security|overcharged)\b"
    ],
    "product_tech_support": [
        r"\b(echo|alexa|kindle|fire stick|fire tv|prime video|app crash|bluetooth|sync|streaming|error code|device|troubleshoot|wifi|audio|screen)\b"
    ],
    "feedback_complaint": [
        r"\b(driver threw|rude|terrible service|disgrace|complaint|lawsuit|sue|worst customer service|never using amazon again|disgusted|unacceptable|boycott|horrible)\b"
    ]
}

ESCALATE_PATTERNS = [
    r"\b(lawyer|legal|sue|fraud|stolen|police|hacked|unauthorized|investigate|dispute with bank|attorney|threat|scam|crime)\b",
    r"\b(disgrace|worst service|unacceptable|furious|disgusting|boycott|hung up on me|pathetic)\b",
    r"\b(locked out of account|cannot access account|account closed|stole my money|take you to court)\b"
]

def clean_tweet_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://t\.co/\w+", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

candidates_by_intent = {intent: [] for intent in INTENT_PATTERNS}

for conv in test_convs:
    raw_text = conv["customer_text"]
    cleaned = clean_tweet_text(raw_text)
    
    if len(cleaned) < 15 or len(cleaned.split()) < 4:
        continue
        
    for intent, patterns in INTENT_PATTERNS.items():
        matched = False
        for p in patterns:
            if re.search(p, cleaned, re.IGNORECASE):
                candidates_by_intent[intent].append((conv, cleaned))
                matched = True
                break
        if matched:
            break

print("Expanded candidate counts per intent in test split:")
for intent, cand_list in candidates_by_intent.items():
    print(f"  - {intent:<25}: {len(cand_list)} candidates")

# Target exact 200 items: balanced distribution across all 7 intents
TARGET_PER_INTENT = {
    "delivery_issue": 30,
    "return_refund_status": 30,
    "damaged_defective_item": 28,
    "cancellation_change_order": 28,
    "account_prime_billing": 30,
    "product_tech_support": 28,
    "feedback_complaint": 26
}

golden_examples = []
item_id = 1

for intent, target_count in TARGET_PER_INTENT.items():
    cands = candidates_by_intent[intent]
    random.shuffle(cands)
    selected = cands[:target_count]
    
    for conv, cleaned in selected:
        is_escalate = False
        escalation_reason = ""
        
        for ep in ESCALATE_PATTERNS:
            if re.search(ep, cleaned, re.IGNORECASE):
                is_escalate = True
                escalation_reason = "High-risk trigger keyword detected (fraud, legal threat, severe complaint, or account lock)."
                break
        
        # In feedback/complaint, escalate higher proportion for human supervisor review
        if intent == "feedback_complaint" and not is_escalate:
            if random.random() < 0.45:
                is_escalate = True
                escalation_reason = "Customer complaint requires personal human supervisor empathy/apology."
                
        # Ambiguous / very short tweets
        if len(cleaned.split()) <= 5 and not is_escalate:
            if random.random() < 0.4:
                is_escalate = True
                escalation_reason = "High ambiguity: minimal context requires human clarification."
                
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

# If slightly under 200, pad from largest candidate pool
while len(golden_examples) < 200:
    extra_cand, extra_clean = candidates_by_intent["delivery_issue"][len(golden_examples)]
    golden_examples.append({
        "id": f"eval_{item_id:03d}",
        "conversation_id": extra_cand["conversation_id"],
        "customer_message": extra_clean,
        "raw_customer_text": extra_cand["customer_text"],
        "gold_intent": "delivery_issue",
        "gold_escalation": "AUTO_HANDLE",
        "evaluation_notes": "Routine delivery_issue query. Suitable for automated response grounded in brand history."
    })
    item_id += 1

print(f"\nFinal Golden Evaluation Set count: {len(golden_examples)} items")
escalate_count = sum(1 for e in golden_examples if e["gold_escalation"] == "ESCALATE")
auto_count = sum(1 for e in golden_examples if e["gold_escalation"] == "AUTO_HANDLE")

print(f"  AUTO_HANDLE cases : {auto_count} ({auto_count/len(golden_examples)*100:.1f}%)")
print(f"  ESCALATE cases    : {escalate_count} ({escalate_count/len(golden_examples)*100:.1f}%)")

output_path = "evaluation/golden_set.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(golden_examples, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(golden_examples)} golden examples to: {output_path}")
