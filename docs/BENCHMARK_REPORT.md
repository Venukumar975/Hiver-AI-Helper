# Hiver AI Support Agent - Automated Evaluation Benchmark Report

- **Generated At**: 2026-09-14 02:09:03
- **Total Runtime**: 51.4 seconds
- **Golden Evaluation Set**: 200 samples

## 1. Intent Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| majority_baseline | 12.50% | 0.0278 | 0.0278 |
| tfidf_baseline | 60.50% | 0.6066 | 0.6066 |
| main_embedding_model | 67.50% | 0.6706 | 0.6706 |

## 2. Retrieval Evaluation (FAISS)

| Metric | Naive Semantic Search | Intent-Aware Retrieval |
| :--- | :---: | :---: |
| Recall@1 | 42.00% | 100.00% |
| Recall@3 | 66.50% | 100.00% |
| MRR | 0.5558 | 1.0000 |

## 3. Escalation Decision Gatekeeper

- **Total Automation Rate**: 59.5%
- **Total Escalation Rate**: 40.5%
- **False Auto-Handle Rate (Missed Risks)**: 3.00%
- **False Escalation Rate**: 34.50%
