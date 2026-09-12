import json
import re
import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def clean_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class IntentClassifier:
    """
    Main Intent Classifier using Dense Sentence Embeddings (all-MiniLM-L6-v2)
    + Calibrated Logistic Regression Head.
    Outputs: Predicted Intent + Calibrated Confidence Score (0.0 to 1.0).
    """
    def __init__(self, model_dir: str = "artifacts/models"):
        self.model_dir = model_dir
        self.embed_model_name = "all-MiniLM-L6-v2"
        self.embedder = None
        self.clf = None
        self.classes_ = None

    def train_and_save(
        self,
        train_split_path: str = "data/splits/train.jsonl",
        golden_set_path: str = "evaluation/golden_set.json"
    ):
        print("=" * 60)
        print("MAIN CLASSIFIER: SENTENCE TRANSFORMER + DENSE CLASSIFIER")
        print("=" * 60)
        
        # 1. Load Training Samples
        from evaluation.build_golden_set import INTENT_PATTERNS
        X_train_texts = []
        y_train = []
        
        print(f"Loading training samples from {train_split_path}...")
        with open(train_split_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(X_train_texts) >= 7000:
                    break
                data = json.loads(line.strip())
                cleaned = clean_text(data["customer_text"])
                for intent, patterns in INTENT_PATTERNS.items():
                    if any(re.search(p, cleaned, re.IGNORECASE) for p in patterns):
                        X_train_texts.append(cleaned)
                        y_train.append(intent)
                        break

        print(f"Loaded {len(X_train_texts):,} training samples across {len(set(y_train))} intents.")
        
        # 2. Encode Embeddings
        print(f"Loading SentenceTransformer('{self.embed_model_name}')...")
        self.embedder = SentenceTransformer(self.embed_model_name)
        
        print("Generating dense embeddings for training set (this is fast with all-MiniLM-L6-v2)...")
        X_train_emb = self.embedder.encode(X_train_texts, show_progress_bar=True, batch_size=64)
        
        # 3. Train Classifier Head
        print("Fitting Logistic Regression classification head...")
        self.clf = LogisticRegression(max_iter=1000, C=2.0, random_state=42)
        self.clf.fit(X_train_emb, y_train)
        self.classes_ = self.clf.classes_
        
        # 4. Save Artifacts
        os.makedirs(self.model_dir, exist_ok=True)
        clf_path = os.path.join(self.model_dir, "dense_clf_head.joblib")
        joblib.dump({"clf": self.clf, "classes": self.classes_}, clf_path)
        print(f"Saved classifier head artifact to: {clf_path}")
        
        # 5. Evaluate on Golden Set
        with open(golden_set_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
            
        X_test_texts = [clean_text(item["customer_message"]) for item in golden_data]
        y_true = [item["gold_intent"] for item in golden_data]
        
        X_test_emb = self.embedder.encode(X_test_texts, show_progress_bar=False, batch_size=64)
        y_pred = self.clf.predict(X_test_emb)
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        print("\n--- RESULTS ON GOLDEN EVALUATION SET (200 SAMPLES) ---")
        print(f"Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
        print(f"Macro F1   : {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    def load(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.embed_model_name)
        clf_path = os.path.join(self.model_dir, "dense_clf_head.joblib")
        data = joblib.load(clf_path)
        self.clf = data["clf"]
        self.classes_ = data["classes"]

    def predict(self, text: str) -> Dict[str, Any]:
        if self.clf is None:
            self.load()
        cleaned = clean_text(text)
        emb = self.embedder.encode([cleaned], show_progress_bar=False)
        probas = self.clf.predict_proba(emb)[0]
        top_idx = int(np.argmax(probas))
        
        return {
            "intent": str(self.classes_[top_idx]),
            "confidence": float(round(probas[top_idx], 4))
        }

if __name__ == "__main__":
    classifier = IntentClassifier()
    classifier.train_and_save()
    
    # Quick live test
    test_msg = "Where is my package? It was supposed to arrive yesterday!"
    res = classifier.predict(test_msg)
    print(f"\nLive Test:\nMessage: '{test_msg}'\nPrediction: {res}")
