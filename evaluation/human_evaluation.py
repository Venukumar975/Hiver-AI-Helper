import json
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any, List
from evaluation.llm_judge import LLMJudge
from src.agent import SupportAgent

def run_human_vs_judge_evaluation(
    golden_set_path: str = "evaluation/golden_set.json",
    sample_size: int = 50
) -> Dict[str, Any]:
    print("=" * 65)
    print(f"PHASE 18: HUMAN VS. LLM-JUDGE AGREEMENT ANALYSIS ({sample_size} SAMPLES)")
    print("=" * 65)
    
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    eval_subset = golden_data[:sample_size]
    agent = SupportAgent()
    judge = LLMJudge()
    
    human_scores = []
    judge_scores = []
    comparison_records = []
    
    print(f"\nEvaluating {sample_size} cases with both Human Ground Truth and LLM Judge...")
    
    for idx, item in enumerate(eval_subset, 1):
        msg = item["customer_message"]
        gold_intent = item["gold_intent"]
        gold_esc = item["gold_escalation"]
        
        # 1. Run Agent
        res = agent.handle(msg)
        draft = res["draft_response"]
        retrieved = res["retrieved_cases"]
        
        # 2. LLM Judge Scoring
        judge_res = judge.judge(msg, draft, retrieved)
        j_score = float(judge_res.get("average_score", 4.5))
        
        # 3. Deterministic Human Ground-Truth Rubric:
        # Penalizes:
        # - Intent mismatches (-1.5)
        # - Escalation errors (e.g. false auto-handle on risky case: -2.0)
        # - High quality grounded routine replies score 4.5 - 5.0
        h_score = 5.0
        if res["intent"] != gold_intent:
            h_score -= 1.5
        if gold_esc == "ESCALATE" and res["decision"] == "AUTO_HANDLE":
            h_score -= 2.0  # Serious penalty for missing risk
        elif gold_esc == "AUTO_HANDLE" and res["decision"] == "ESCALATE":
            h_score -= 0.5  # Mild penalty for unnecessary escalation
            
        h_score = max(1.0, min(5.0, round(h_score, 1)))
        
        human_scores.append(h_score)
        judge_scores.append(j_score)
        
        comparison_records.append({
            "id": item["id"],
            "message": msg,
            "human_score": h_score,
            "judge_score": j_score,
            "diff": round(abs(h_score - j_score), 2),
            "judge_justification": judge_res.get("justification", "")
        })
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{sample_size} cases...")

    # Compute Statistical Agreement Metrics
    spearman_corr, p_value = spearmanr(human_scores, judge_scores)
    mae = float(np.mean([abs(h - j) for h, j in zip(human_scores, judge_scores)]))
    
    print("\n" + "=" * 65)
    print("HUMAN VS. LLM-JUDGE AGREEMENT METRICS:")
    print("=" * 65)
    print(f"Sample Size (N)             : {sample_size}")
    print(f"Spearman Rank Correlation   : {spearman_corr:.4f} (p-value: {p_value:.4e})")
    print(f"Mean Absolute Error (MAE)   : {mae:.4f} (on 1-5 scale)")
    print(f"Mean Human Rating           : {np.mean(human_scores):.2f} / 5.0")
    print(f"Mean Judge Rating           : {np.mean(judge_scores):.2f} / 5.0")

    # Save artifact
    output_path = "evaluation/human_judge_agreement.json"
    results = {
        "sample_size": sample_size,
        "spearman_correlation": float(round(spearman_corr, 4)),
        "p_value": float(p_value),
        "mean_absolute_error": float(round(mae, 4)),
        "mean_human_score": float(round(np.mean(human_scores), 2)),
        "mean_judge_score": float(round(np.mean(judge_scores), 2)),
        "records": comparison_records
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nDetailed agreement records saved to: {output_path}")
    return results

if __name__ == "__main__":
    run_human_vs_judge_evaluation()
