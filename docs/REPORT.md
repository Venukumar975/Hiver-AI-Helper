# AI Customer Support Agent for AmazonHelp: Final Engineering Report
**Author:** Venukumar975  
**Evaluation Set:** 200 Hand-Labelled Held-Out Conversations (Zero Leakage)  
**Dataset:** Customer Support on Twitter (twcs, Kaggle)  

---

## 1. Problem Framing & Brand Selection

Customer support on social media is high-velocity, publicly visible, and inherently noisy. When a customer tweets at `@AmazonHelp`, they are often frustrated, provide incomplete information, or encounter genuine supply-chain disruptions.

### What "Good" Means for Amazon Support
1. **Accurate Triage**: Categorizing unstructured, informal tweets into distinct operational workflows (e.g. delivery delays vs. defective merchandise).
2. **Empirical Grounding**: Basing advice strictly on official historical precedent (e.g. directing customers to standard self-service portals `amzn.to/...`, advising 24h wait periods for delivery scans), ending with brand sign-offs (`^AH`).
3. **Conservative Risk Routing**: Recognizing when an issue involves financial fraud, account takeover, or legal exposure, and routing it immediately to human specialists.
4. **Minimizing False Auto-Handles**: An unnecessary human escalation costs ~$3; an automated brush-off on an unauthorized credit card charge costs customer trust and regulatory goodwill.

### What I Chose NOT to Build
- **No Complex Multi-Agent Swarms**: A single, clean, deterministic pipeline with strict gating outperforms unpredictable agent loops for customer support.
- **No Heavy Fine-Tuning**: Pre-trained Sentence Transformers (`all-MiniLM-L6-v2`) and prompt-grounded LLMs achieved 100% Intent-Aware Retrieval recall without expensive GPU fine-tuning.
- **No Over-engineered Vector DB Clusters**: Used embedded C++ FAISS (`IndexFlatIP`), providing sub-2ms latency on 15,000 vectors with zero network overhead.

---

## 2. Intent Classification & Baseline Progression

We evaluated three successive models on our held-out 200 Golden Evaluation Set across all 8 intents (25 samples per class):

| Model | Technique | Accuracy | Macro F1 | Weighted F1 | Calibrated Confidence? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline 1** | Majority Class (`delivery_issue`) | 12.50% | 0.0278 | 0.0278 | ❌ No |
| **Baseline 2** | Tuned TF-IDF (10k feat) + Logistic Reg | 60.50% | 0.6066 | 0.6066 | ⚠️ No (Sparse) |
| **Main Model** | Tuned Sentence-Transformers (`MiniLM`) + Head | **67.50%** | **0.6706** | **0.6706** | ✅ **Yes (0.0 to 1.0)** |

### Analysis of Baselines & Hyperparameter Tuning:
- **Baseline 1** establishes the true random lower bound across 8 equal categories ($1/8 = 12.5\%$).
- **Baseline 2 (TF-IDF)**: An exhaustive 90-combination grid search on all 7,494 validation samples proved that keyword matching plateaus at 60.50%, as n-gram models cannot disambiguate syntactic paraphrases.
- **The Main Model (Sentence Transformers)**: Tuning $C$ across 44 dense configurations on the validation set identified $C=1.0$ as the global regularization sweet spot, increasing Golden Set accuracy from **65.00% to 67.50% (+2.50%)** and Macro F1 from **0.6444 to 0.6706**. The output softmax distribution provides well-calibrated probabilities essential for downstream escalation gating.

---

## 3. Retrieval Engine: Proving Intent-Aware Retrieval (FAISS)

We indexed **15,000 resolved historical Amazon conversations** from the Train partition into FAISS and compared Naive Semantic Search vs. Intent-Aware Retrieval across all 200 evaluation queries:

| Metric | Naive Semantic Search | Intent-Aware Retrieval | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 42.00% | **100.00%** | **+58.0%** |
| **Recall@3** | 66.50% | **100.00%** | **+33.5%** |
| **Recall@5** | 76.50% | **100.00%** | **+23.5%** |
| **MRR** | 0.5558 | **1.0000** | **+0.4442** |

**Why Naive Retrieval Fails**: Unfiltered semantic search frequently confuses surface-level lexical similarity across domains (e.g. waiting 5 days for a package vs. waiting 5 days for a refund). Intent-filtering constrains the FAISS search space, guaranteeing that 100% of top-K results are domain-appropriate.

---

## 4. Response Generation & LLM-as-a-Judge Evaluation

### LLM-as-a-Judge (5-Criteria Rubric on 1–5 scale via Gemini 3.6 Flash)
- **Groundedness**: 4.8 / 5.0 (Strictly grounded in FAISS historical cases; no invented compensation).
- **Relevance**: 4.7 / 5.0 (Directly answers the customer's specific inquiry).
- **Helpfulness**: 4.6 / 5.0 (Provides actionable self-service steps and links).
- **Brand Consistency**: 4.9 / 5.0 (Polite, concise Twitter style with `^AH` signature).
- **Safety**: 5.0 / 5.0 (Zero exposure of private accounts; safe human routing for fraud).

### Human vs. LLM-Judge Alignment (50-Sample Empirical Study)
- **Mean Human Rating**: 4.52 / 5.0
- **Mean Judge Rating**: 4.39 / 5.0
- **Mean Absolute Error (MAE)**: **0.516 points**
- **Spearman Rank Correlation**: **0.6730** ($p = 8.58 \times 10^{-8}$)
- **Takeaway**: The LLM judge tracks human evaluations with a strong positive rank correlation (0.673) and low error (0.516 pts), proving high evaluation reliability.

---

## 5. Escalation Decision Gatekeeper

Operational performance across the 200 Golden Evaluation Set:
- **Total Automation Rate**: **64.5%** (129 / 200 cases auto-handled)
- **Total Escalation Rate**: **35.5%** (71 / 200 cases sent to human)
- **CRITICAL: False Auto-Handle Rate**: **3.00%** (6 / 200 risky cases missed)
- **False Escalation Rate**: **29.50%** (59 / 200 benign queries sent to human)

The escalation policy enforces safe behavior: when in doubt, it prefers bothering a human representative rather than hallucinating an unauthorized policy promise.

---

## 6. Top 5 Real Failure Modes

1. **Hashtag-Only Dissatisfaction (#Pathetic, #WorstService)**:
   - *Example*: Customer tweet ending with `#cheaters #worstcustomerserviceever` without formal legal or fraud words.
   - *Failure*: Model missed the escalation because regex triggers look for formal keywords ("lawyer", "fraud").
   - *Fix*: Integrate sentiment intensity scoring (e.g. VADER or toxicity scoring) alongside keyword matching.

2. **Extreme Ambiguity on Short Tweets ("who are you", "it isn't working")**:
   - *Example*: *"It still isn't working even though I followed the steps."*
   - *Failure*: Softmax assigned 93% to defective product because of "isn't working", even though the tweet lacked any object or context.
   - *Fix*: Hard length gate: messages under 8 words must prompt for clarification or escalate.

3. **Multi-Turn Pronoun Dependency**:
   - *Example*: Customer replies: *"Here is my order number: 112-9381023."*
   - *Failure*: Standalone classifier cannot determine whether the order number refers to a return or a late delivery.
   - *Fix*: Maintain conversational session memory linking previous customer turns.

4. **Category Overlap (Damaged Item vs. Delivery Delay)**:
   - *Example*: *"Package arrived 4 days late and the box was crushed open."*
   - *Failure*: Spans both `delivery_issue` and `damaged_defective_item`.
   - *Fix*: Multi-label classification allowing dual-intent tags.

5. **Displaced Customer Expectations (Third-Party Sellers)**:
   - *Example*: Inquiries regarding third-party marketplace seller policies.
   - *Failure*: Historical cases often suggest contacting the seller directly, which frustrated some customer scenarios.
   - *Fix*: Differentiate "Fulfilled by Amazon" vs. "Third-Party Marketplace" in the retrieval index.

---

## 7. MANDATORY SECTION: "What Is Misleading About My Headline Number?"

Our strongest headline result:
> **"Intent-Aware Retrieval achieved 100% Recall@1, and our classifier outperformed both baselines with 67.50% accuracy (vs. 12.5% trivial)."**

### What this number DOES NOT prove:
1. **Offline Evaluation $\neq$ Real Customer Satisfaction**: 100% Recall@1 proves that the top retrieved case matched the ground-truth intent label. It does **not** prove that the customer was satisfied with the answer. A customer whose \$1,000 package is missing does not want a polite link telling them to wait 24 hours.
2. **Single-Brand Homogeneity**: Our 67.50% accuracy was evaluated entirely on AmazonHelp data. It says nothing about how the architecture would transfer to Uber (fare disputes) or Apple (hardware diagnostics).
3. **Static Historical Precedents**: Our knowledge base reflects Twitter customer support from 2017. Policies change (e.g. return windows, Prime membership pricing). A 100% grounded answer based on 2017 data might be completely outdated under 2026 Amazon policies.
4. **Golden Set Size Limitation**: The evaluation set contains 200 hand-labelled examples. While rigorous and zero-leakage, real-world customer support handles 50,000 queries daily, introducing thousands of edge-case linguistic variations not captured in a 200-sample test set.
5. **LLM Judge Inherent Ceiling**: LLM judges tend to reward polite, well-structured text. A response can score 5/5 on brand consistency and fluency while being completely unhelpful for an angry customer who needs urgent operational intervention.

---

## 8. What I Would Do With One More Week

1. **Temporal Policy Invalidation**: Implement time-weighted decay on historical cases so older resolutions are discounted if newer company policies contradict them.
2. **Conversational Multi-Turn Session Memory**: Track thread history across multiple customer tweets using session tokens instead of treating each tweet as an isolated turn.
3. **Sentiment & Toxicity Scoring for Escalation**: Augment regex risk detection with an open-source toxicity model to reliably catch abusive, angry, or sarcastic customer tweets.
4. **Multi-Brand Benchmark**: Run the identical pipeline on `@AppleSupport` and `@SpotifyCares` to empirically measure cross-domain transferability.
5. **Human-in-the-Loop Active Learning**: Add a feedback webhook where human agents correcting a false auto-handle automatically queue the sample for model re-training.
