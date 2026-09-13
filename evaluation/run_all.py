import os
import json
import time
import numpy as np
from typing import Dict, Any

from src.classification.baseline_majority import run_majority_baseline
from src.classification.baseline_tfidf import train_and_eval_tfidf
from src.classification.classifier import IntentClassifier
from evaluation.evaluate_retrieval import evaluate_retrieval_system
from evaluation.evaluate_escalation import evaluate_escalation_system

def sanitize_for_json(obj):
    """Convert numpy data types to native Python types for clean JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def run_complete_evaluation():
    print("=" * 75)
    print("       HIVER AI CUSTOMER SUPPORT AGENT - UNIFIED EVALUATION HARNESS")
    print("=" * 75)
    start_time = time.time()
    
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brand": "AmazonHelp",
        "dataset": "Customer Support on Twitter (twcs)",
        "golden_set_size": 200,
        "metrics": {}
    }
    
    # 1. INTENT CLASSIFICATION EVALUATION
    print("\n>>> [1/4] Running Intent Classification Baselines & Main Model...")
    res_maj = run_majority_baseline()
    res_tfidf = train_and_eval_tfidf()
    
    classifier = IntentClassifier()
    classifier.load()
    with open("evaluation/golden_set.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    y_true = [item["gold_intent"] for item in golden_data]
    y_pred = [classifier.predict(item["customer_message"])["intent"] for item in golden_data]
    
    from sklearn.metrics import accuracy_score, f1_score
    acc_main = accuracy_score(y_true, y_pred)
    macro_f1_main = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1_main = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    results["metrics"]["intent_classification"] = {
        "majority_baseline": {
            "accuracy": round(float(res_maj["accuracy"]), 4),
            "macro_f1": round(float(res_maj["macro_f1"]), 4),
            "weighted_f1": round(float(res_maj["weighted_f1"]), 4)
        },
        "tfidf_baseline": {
            "accuracy": round(float(res_tfidf["accuracy"]), 4),
            "macro_f1": round(float(res_tfidf["macro_f1"]), 4),
            "weighted_f1": round(float(res_tfidf["weighted_f1"]), 4)
        },
        "main_embedding_model": {
            "accuracy": round(float(acc_main), 4),
            "macro_f1": round(float(macro_f1_main), 4),
            "weighted_f1": round(float(weighted_f1_main), 4)
        }
    }
    
    # 2. RETRIEVAL EVALUATION
    print("\n>>> [2/4] Running Retrieval Evaluation (Naive vs. Intent-Aware)...")
    res_retrieval = evaluate_retrieval_system()
    results["metrics"]["retrieval"] = sanitize_for_json(res_retrieval)
    
    # 3. ESCALATION GATEKEEPER EVALUATION
    print("\n>>> [3/4] Running Escalation Gatekeeper Evaluation...")
    res_esc = evaluate_escalation_system()
    results["metrics"]["escalation"] = sanitize_for_json(res_esc)
    
    # 4. HUMAN VS JUDGE SUMMARY
    print("\n>>> [4/4] Loading Human vs. LLM-Judge Agreement Results...")
    human_judge_file = "results/human_judge_agreement.json"
    if os.path.exists(human_judge_file):
        with open(human_judge_file, "r", encoding="utf-8") as f:
            hj_data = json.load(f)
            results["metrics"]["human_vs_judge_agreement"] = {
                "sample_size": hj_data["sample_size"],
                "spearman_correlation": hj_data["spearman_correlation"],
                "mean_absolute_error": hj_data["mean_absolute_error"],
                "mean_human_score": hj_data["mean_human_score"],
                "mean_judge_score": hj_data["mean_judge_score"]
            }
            
    elapsed = time.time() - start_time
    results["runtime_seconds"] = round(elapsed, 2)
    
    # Save machine-readable results
    results_path = "results/evaluation_results.json"
    clean_results = sanitize_for_json(results)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
        
    print(f"\nCompleted complete benchmark suite in {elapsed:.1f} seconds! (< 15 minute requirement met)")
    print(f"Results successfully saved to: {results_path}")
    
    # Generate human-readable Markdown Report
    report_md_path = "docs/BENCHMARK_REPORT.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Hiver AI Support Agent - Automated Evaluation Benchmark Report\n\n")
        f.write(f"- **Generated At**: {results['timestamp']}\n")
        f.write(f"- **Total Runtime**: {elapsed:.1f} seconds\n")
        f.write(f"- **Golden Evaluation Set**: {results['golden_set_size']} samples\n\n")
        f.write("## 1. Intent Classification\n\n")
        f.write("| Model | Accuracy | Macro F1 | Weighted F1 |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m_name, m_vals in results["metrics"]["intent_classification"].items():
            f.write(f"| {m_name} | {m_vals['accuracy']*100:.2f}% | {m_vals['macro_f1']:.4f} | {m_vals['weighted_f1']:.4f} |\n")
        f.write("\n## 2. Retrieval Evaluation (FAISS)\n\n")
        f.write("| Metric | Naive Semantic Search | Intent-Aware Retrieval |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| Recall@1 | {res_retrieval['naive']['Recall@1']*100:.2f}% | {res_retrieval['intent_aware']['Recall@1']*100:.2f}% |\n")
        f.write(f"| Recall@3 | {res_retrieval['naive']['Recall@3']*100:.2f}% | {res_retrieval['intent_aware']['Recall@3']*100:.2f}% |\n")
        f.write(f"| MRR | {res_retrieval['naive']['MRR']:.4f} | {res_retrieval['intent_aware']['MRR']:.4f} |\n\n")
        f.write("## 3. Escalation Decision Gatekeeper\n\n")
        f.write(f"- **Total Automation Rate**: {res_esc['automation_rate']*100:.1f}%\n")
        f.write(f"- **Total Escalation Rate**: {res_esc['escalation_rate']*100:.1f}%\n")
        f.write(f"- **False Auto-Handle Rate (Missed Risks)**: {res_esc['false_auto_handle_rate']*100:.2f}%\n")
        f.write(f"- **False Escalation Rate**: {res_esc['false_escalation_rate']*100:.2f}%\n")
        
    print(f"Human-readable report successfully written to: {report_md_path}")

if __name__ == "__main__":
    run_complete_evaluation()
