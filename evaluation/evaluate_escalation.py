import json
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.escalation.policy import EscalationPolicy
from src.classification.classifier import IntentClassifier
from src.retrieval.retriever import HistoricalCaseRetriever

def evaluate_escalation_system(
    golden_set_path: str = "evaluation/golden_set.json"
) -> Dict[str, Any]:
    print("=" * 65)
    print("PHASE 11 & 19: ESCALATION GATEKEEPER EVALUATION")
    print("=" * 65)
    
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    classifier = IntentClassifier()
    retriever = HistoricalCaseRetriever()
    policy = EscalationPolicy()
    
    y_true = []
    y_pred = []
    reasons = []
    
    false_auto_handles = []
    false_escalations = []
    
    print(f"Running end-to-end escalation checks on {len(golden_data)} Golden Set samples...")
    for item in golden_data:
        gold_decision = item["gold_escalation"]  # "AUTO_HANDLE" or "ESCALATE"
        msg = item["customer_message"]
        
        # 1. Classify intent & confidence
        clf_res = classifier.predict(msg)
        pred_intent = clf_res["intent"]
        conf = clf_res["confidence"]
        
        # 2. Stage 1: Pre-retrieval check (confidence & risk triggers)
        pre_esc = policy.evaluate_pre_retrieval(msg, pred_intent, conf)
        if pre_esc["decision"] == "ESCALATE":
            pred_decision = "ESCALATE"
            esc_reason = pre_esc["reason"]
            retrieved = []
        else:
            # 3. Stage 2: Intent-aware retrieval from FAISS
            retrieved = retriever.retrieve_intent_aware(msg, intent=pred_intent, top_k=3)
            post_esc = policy.evaluate_retrieval_quality(msg, pred_intent, conf, retrieved)
            pred_decision = post_esc["decision"]
            esc_reason = post_esc["reason"]
        
        y_true.append(gold_decision)
        y_pred.append(pred_decision)
        reasons.append(esc_reason)
        
        # Track errors
        if gold_decision == "ESCALATE" and pred_decision == "AUTO_HANDLE":
            false_auto_handles.append({
                "id": item["id"],
                "message": msg,
                "notes": item["evaluation_notes"],
                "predicted_intent": pred_intent,
                "confidence": conf
            })
        elif gold_decision == "AUTO_HANDLE" and pred_decision == "ESCALATE":
            false_escalations.append({
                "id": item["id"],
                "message": msg,
                "reason": esc_reason
            })
            
    # Binary metrics (treating ESCALATE as positive class 1)
    y_true_bin = [1 if d == "ESCALATE" else 0 for d in y_true]
    y_pred_bin = [1 if d == "ESCALATE" else 0 for d in y_pred]
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    total = len(y_true)
    tn, fp, fn, tp = cm.ravel()
    
    auto_rate = (tn + fn) / total
    esc_rate = (tp + fp) / total
    false_auto_rate = fn / total
    false_esc_rate = fp / total
    
    print("\n--- ESCALATION METRICS ON GOLDEN SET (200 SAMPLES) ---")
    print(f"Accuracy                 : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (ESCALATE)     : {prec:.4f}")
    print(f"Recall (ESCALATE)        : {rec:.4f} ({tp}/{tp+fn} dangerous/complex cases caught)")
    print(f"F1-Score (ESCALATE)      : {f1:.4f}")
    print(f"\n--- OPERATIONAL TRADE-OFFS ---")
    print(f"Total Automation Rate    : {auto_rate*100:.1f}% ({tn+fn}/{total})")
    print(f"Total Escalation Rate    : {esc_rate*100:.1f}% ({tp+fp}/{total})")
    print(f"CRITICAL: False Auto-Rate: {false_auto_rate*100:.2f}% ({fn}/{total} risky issues missed)")
    print(f"False Escalation Rate    : {false_esc_rate*100:.2f}% ({fp}/{total} benign queries sent to human)")
    
    print("\nConfusion Matrix:")
    print(f"                 Pred AUTO    Pred ESCALATE")
    print(f"Actual AUTO   :    {tn:<10}   {fp:<10}")
    print(f"Actual ESCALATE:   {fn:<10}   {tp:<10}")
    
    if false_auto_handles:
        print("\n--- SAMPLE FALSE AUTO-HANDLES (MISSED RISKS) ---")
        for f_ah in false_auto_handles[:2]:
            print(f"[{f_ah['id']}] Message: '{f_ah['message']}'")
            print(f"  True Risk: {f_ah['notes']}\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "automation_rate": auto_rate,
        "escalation_rate": esc_rate,
        "false_auto_handle_count": fn,
        "false_auto_handle_rate": false_auto_rate,
        "false_escalation_count": fp,
        "false_escalation_rate": false_esc_rate
    }

if __name__ == "__main__":
    evaluate_escalation_system()
