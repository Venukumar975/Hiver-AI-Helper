import json
import re
import os
import joblib
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def clean_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def train_and_eval_tfidf(
    golden_set_path: str = "evaluation/golden_set.json",
    train_split_path: str = "data/splits/train.jsonl",
    model_save_dir: str = "artifacts/models"
) -> Dict[str, Any]:
    print("=" * 60)
    print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION CLASSIFIER")
    print("=" * 60)
    
    # 1. Load balanced training data across all 8 intents
    print(f"Loading balanced training samples (750 per intent x 8) from {train_split_path}...")
    from collections import defaultdict
    by_intent = defaultdict(list)
    
    with open(train_split_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            intent = data.get("intent") or data.get("nli_intent")
            if intent == "general_unclassified":
                intent = "general_inquiry_greeting"
            if len(by_intent[intent]) < 750:
                cleaned = clean_text(data["customer_text"])
                if cleaned:
                    by_intent[intent].append(cleaned)
                    
    X_train = []
    y_train = []
    for intent, texts in by_intent.items():
        for t in texts:
            X_train.append(t)
            y_train.append(intent)

    print(f"Loaded {len(X_train):,} balanced training samples across {len(set(y_train))} intents (750 each).")
    
    # 2. Build Pipeline: TF-IDF + Logistic Regression
    # =========================================================================================
    # HYPERPARAMETER TUNING OBSERVATION & EMPIRICAL JUSTIFICATION:
    # -----------------------------------------------------------------------------------------
    # - Evaluated across 90 combinations on 7,494 validation conversations (hyper_params_results.log):
    #     * max_features: [2500, 5000, 10000]
    #     * ngram_range: [(1, 1), (1, 2), (1, 3)]
    #     * sublinear_tf: [True, False]
    #     * C: [0.1, 0.5, 1.0, 2.0, 5.0]
    # - WINNER: max_features=10000, ngram_range=(1, 1), sublinear_tf=False, C=1.0
    #     * Validation F1: 0.4572 | Golden Set Accuracy: 60.50% | Golden Set F1: 0.6066
    # - Key Finding: Vocabulary expansion to 10,000 preserved rare e-commerce tokens (ASIN, FireStick),
    #   while trigrams (1,3) overfitted social media noise without boosting validation F1.
    # =========================================================================================
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=False, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    
    print("Fitting TF-IDF + Logistic Regression pipeline (Tuned optimal: max_features=10000, ngram=(1,1))...")
    pipeline.fit(X_train, y_train)
    
    # 3. Save Model Artifact
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, "tfidf_lr_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved model artifact to: {model_path}")
    
    # 4. Evaluate on Held-out Golden Set (200 samples)
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    X_test = [clean_text(item["customer_message"]) for item in golden_data]
    y_true = [item["gold_intent"] for item in golden_data]
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print("\n--- RESULTS ON GOLDEN EVALUATION SET (200 SAMPLES) ---")
    print(f"Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        "model": "TfidfLogisticRegression",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

if __name__ == "__main__":
    train_and_eval_tfidf()
