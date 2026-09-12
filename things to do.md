# Hiver SDE Intern Take-Home: Project Requirements & Checklist

> **Core Objective**: Turn a messy real-world customer support dataset into a working AI support agent and **prove it works**. The proof is worth more than the system.

---

## 1. Problem Statement
Pick **one brand** from the *Customer Support on Twitter* dataset and build an AI support agent that can:
1. **Classify** each incoming customer message into a small intent taxonomy derived from data.
2. **Draft a reply** grounded in how that brand has historically resolved similar issues.
3. **Decide** whether the message should be auto-handled or escalated to a human, with a stated reason.
4. **Convince reviewers the agent is reliable** through a rigorous, reproducible evaluation pipeline.

---

## 2. Mandatory Deliverables Checklist

- [ ] **Runnable Pipeline & Repository**:
  - Clean modular structure (`src/`, `evaluation/`, `app/`, `docs/`).
  - Strict 15-minute reproduction requirement for headline numbers.
  - Dependencies clearly documented in `requirements.txt`.
- [ ] **Golden Evaluation Set**:
  - 150–250 hand-labelled examples (target: ~200).
  - Documented sampling strategy and labeling methodology.
  - Complete isolation from training/retrieval corpora (conversation-level split).
- [ ] **Evaluation Harness**:
  - Automated quantitative metrics across Intent, Retrieval, and Escalation.
  - LLM-as-a-judge rubric (1–5 scale) for reply quality (*Groundedness, Relevance, Helpfulness, Brand Consistency, Safety*).
  - Empirical agreement measurement between LLM judge and human ratings (~50 examples).
- [ ] **Comprehensive Report (Max 6 Pages / Markdown)**:
  - Problem framing: what "good" means for this brand, and what you chose not to build.
  - Results vs at least two baselines (Trivial Majority-class + Simple TF-IDF/Logistic Regression).
  - Top 5 failure modes with real examples, hypotheses, and fixes.
  - **Mandatory section**: *"What is misleading about my headline number?"*
  - *"What I'd do next with one more week."*
  - **Decision Log**: 10–15 non-obvious decisions with choices, alternatives, reasons, and consequences.
- [ ] **Lightweight Demo UI**:
  - Streamlit app demonstrating classification, retrieval evidence, grounded draft, and escalation decision.

---

## 3. Detailed Phase Breakdown

### Phase 1 — Dataset Understanding & Brand Selection
- Inspect the Twitter dataset: brands, message counts, conversation threads, resolution indicators.
- Select 1 brand based on conversation volume, customer/brand interaction ratio, and variety of issues.
- Reconstruct threaded conversations (`conversation_id`, `brand`, `messages[]`).
- Filter for useful historical pairs: Customer Problem -> Brand Resolution.
- Document data quality issues (noise, missing turns, DM redirects).

### Phase 2 — Intent Taxonomy Definition
- Inspect a representative sample of customer messages for the chosen brand.
- Derive a small, domain-specific taxonomy (e.g., 6–10 intents).
- Define each intent with inclusion criteria, exclusion criteria, and 5–10 concrete examples.
- Document reasons for splits and merges.

### Phase 3 — Data Splitting & Leakage Prevention
- Perform train/dev/test splitting strictly at the **Conversation level** (e.g., 70% corpus, 15% validation, 15% test).
- Guarantee zero cross-contamination between retrieval corpus and evaluation sets.

### Phase 4 — Golden Evaluation Set (~200 Examples)
- Stratified sampling across intents, difficulty levels, noise, short/long messages, and escalation cases.
- Manually label with gold intent, gold escalation decision (`AUTO_HANDLE` vs `ESCALATE`), and notes.

### Phase 5 — Baseline 1: Trivial Majority Classifier
- Predict the most frequent class for all inputs. Measure Accuracy, Macro F1, Weighted F1, Per-class F1.

### Phase 6 — Baseline 2: Classical ML (TF-IDF + Logistic Regression)
- Train TF-IDF vectorizer + Logistic Regression on training set.
- Evaluate on golden set; generate confusion matrix and classification report.

### Phase 7 — Main Intent Classifier (Embeddings + Classifier)
- Dense sentence embeddings (`sentence-transformers`) + classifier head.
- Output predicted intent and calibrated confidence score.
- Evaluate and compare against Baselines 1 & 2.

### Phase 8 & 9 — Historical Resolution Database & Intent-Aware Retrieval
- Build vector index (FAISS) of resolved historical cases (`customer_issue`, `brand_resolution`, `case_id`, `intent`).
- Implement intent-aware retrieval: filter by predicted intent, then rank by semantic similarity.
- Compare naive semantic search vs intent-filtered search (Recall@1, Recall@3, Recall@5, MRR).

### Phase 10 & 11 — Grounded Response Generation & Conservative Escalation Policy
- Prompt LLM with customer issue, predicted intent, and top-K historical resolutions.
- Require draft response grounded in historical resolutions, citing evidence case IDs.
- Build conservative escalation gatekeeper using confidence thresholds, retrieval similarity, and risk keywords.
- Optimize specifically to minimize **False Auto-Handles**.

### Phase 12 — End-to-End Agent
- Single entrypoint: `agent.handle(message)` returning intent, confidence, retrieved evidence, draft response, decision, and rationale.

### Phase 13–18 — Evaluation Suite & Judge Validation
- Automated evaluation runner: `python -m evaluation.run_all`.
- LLM-as-a-judge 1–5 scoring with justifications.
- Human validation on ~50 samples with Spearman correlation / Cohen's Kappa analysis.

### Phase 19–23 — Failure Analysis, Honest Critique & Decision Log
- Deep dive into top 5 real failure modes.
- Mandatory critique: *"What is misleading about my headline number?"*
- 10–15 non-obvious engineering decisions logged in `docs/DECISIONS.md`.

### Phase 24–26 — Demo UI & 15-Minute Reproducibility
- Streamlit demo showing real-time handling, retrieved evidence, and escalation logic.
- Cache precomputed artifacts/models so external evaluators can run tests in <15 minutes.
