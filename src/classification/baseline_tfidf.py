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
    
    # 1. Load training data (sample 10,000 from train split for swift training)
    print(f"Loading training samples from {train_split_path}...")
    
    # For supervised training of the TF-IDF baseline, we use the regex intent matches on train split
    from evaluation.build_golden_set import INTENT_PATTERNS
    
    X_train = []
    y_train = []
    
    with open(train_split_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(X_train) >= 7000:  # 1,000 balanced per intent approx
                break
            data = json.loads(line.strip())
            cleaned = clean_text(data["customer_text"])
            
            for intent, patterns in INTENT_PATTERNS.items():
                if any(re.search(p, cleaned, re.IGNORECASE) for p in patterns):
                    X_train.append(cleaned)
                    y_train.append(intent)
                    break

    print(f"Loaded {len(X_train):,} training samples across {len(set(y_train))} intents.")
    
    # 2. Build Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    
    print("Fitting TF-IDF + Logistic Regression pipeline...")
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
