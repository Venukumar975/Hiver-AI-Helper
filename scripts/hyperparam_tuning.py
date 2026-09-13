import os
import sys
import json
import re
import time
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))
sys.stdout.reconfigure(line_buffering=True)

def clean_text(text: str) -> str:
    text = re.sub(r"^(@\w+\s*)+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def run_exhaustive_hyperparam_tuning():
    log_file = "hyper_params_results.log"
    print("=" * 90)
    print("     COMPREHENSIVE HYPERPARAMETER TUNING & GRID SEARCH SUITE")
    print("=" * 90)
    start_total = time.time()

    # 1. LOAD TRAINING DATA (6,000 Balanced Samples from Train Split)
    print("\n[Step 1/5] Loading 6,000 balanced training samples (750 per intent x 8)...")
    train_split_path = "data/splits/train.jsonl"
    by_intent_train = defaultdict(list)
    with open(train_split_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            intent = d.get("intent") or d.get("nli_intent")
            if intent == "general_unclassified":
                intent = "general_inquiry_greeting"
            if len(by_intent_train[intent]) < 750:
                cleaned = clean_text(d["customer_text"])
                if cleaned:
                    by_intent_train[intent].append(cleaned)

    X_train = []
    y_train = []
    for intent, texts in by_intent_train.items():
        for t in texts:
            X_train.append(t)
            y_train.append(intent)
    print(f"  -> Loaded {len(X_train):,} training examples across {len(set(y_train))} intents.")

    # 2. LOAD COMPLETE VALIDATION SET (ALL 7,496 SAMPLES)
    print("\n[Step 2/5] Loading COMPLETE Validation Set (all rows from data/splits/val.jsonl)...")
    val_split_path = "data/splits/val.jsonl"
    X_val = []
    y_val = []
    with open(val_split_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            intent = d.get("intent") or d.get("nli_intent")
            if intent == "general_unclassified":
                intent = "general_inquiry_greeting"
            cleaned = clean_text(d["customer_text"])
            if cleaned:
                X_val.append(cleaned)
                y_val.append(intent)
    print(f"  -> Loaded {len(X_val):,} COMPLETE validation examples.")

    # 3. LOAD 200 HELD-OUT GOLDEN SET
    print("\n[Step 3/5] Loading 200 Held-Out Golden Set (evaluation/golden_set.json)...")
    with open("evaluation/golden_set.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    X_gold = [clean_text(item["customer_message"]) for item in golden_data]
    y_gold = [item["gold_intent"] for item in golden_data]
    print(f"  -> Loaded {len(X_gold):,} Golden Set examples.")

    # PREPARE LOG FILE
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write("               HIVER AI AGENT - EXHAUSTIVE HYPERPARAMETER TUNING LOG\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data  : 6,000 balanced samples (750 x 8 intents) from data/splits/train.jsonl\n")
        f.write(f"Validation Data: ALL {len(X_val):,} samples from data/splits/val.jsonl\n")
        f.write(f"Golden Set Data: ALL {len(X_gold):,} samples from evaluation/golden_set.json\n")
        f.write("=" * 130 + "\n\n")

    # 4. GRID SEARCH FOR MODEL 1: TF-IDF + LOGISTIC REGRESSION
    print("\n" + "=" * 90)
    print("SECTION A: GRID SEARCH ON TF-IDF + LOGISTIC REGRESSION")
    print("=" * 90)

    tfidf_grid = []
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        for max_feat in [2500, 5000, 10000]:
            for sublinear in [True, False]:
                for c_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
                    tfidf_grid.append({
                        "ngram_range": ngram,
                        "max_features": max_feat,
                        "sublinear_tf": sublinear,
                        "C": c_val
                    })

    print(f"Total TF-IDF Parameter Combinations to Evaluate: {len(tfidf_grid)}")

    tfidf_results = []
    tfidf_table_header = (
        f"{'Run':<5} | {'N-Gram':<8} | {'Max Feat':<8} | {'Sublinear':<9} | {'C':<5} | "
        f"{'Val Acc':<8} | {'Val F1(M)':<9} | {'Gold Acc':<8} | {'Gold F1(M)':<10} | {'Time(s)':<7}\n"
    )
    tfidf_table_sep = "-" * 115 + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("### SECTION A: TF-IDF + LOGISTIC REGRESSION HYPERPARAMETER RESULTS\n")
        f.write(tfidf_table_header)
        f.write(tfidf_table_sep)

    for idx, params in enumerate(tfidf_grid, 1):
        t0 = time.time()
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=params["ngram_range"],
                max_features=params["max_features"],
                sublinear_tf=params["sublinear_tf"],
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                C=params["C"],
                max_iter=1000,
                random_state=42
            ))
        ])

        pipeline.fit(X_train, y_train)

        # Predict on ALL 7,496 Val
        val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_f1_weighted = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        # Predict on 200 Golden Set
        gold_pred = pipeline.predict(X_gold)
        gold_acc = accuracy_score(y_gold, gold_pred)
        gold_f1_macro = f1_score(y_gold, gold_pred, average="macro", zero_division=0)

        elapsed = time.time() - t0

        record = {
            "run": idx,
            "params": params,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "gold_acc": gold_acc,
            "gold_f1_macro": gold_f1_macro,
            "elapsed": elapsed
        }
        tfidf_results.append(record)

        log_line = (
            f"T{idx:02d}   | {str(params['ngram_range']):<8} | {params['max_features']:<8} | "
            f"{str(params['sublinear_tf']):<9} | {params['C']:<5.1f} | "
            f"{val_acc*100:<7.2f}% | {val_f1_macro:<9.4f} | {gold_acc*100:<7.2f}% | {gold_f1_macro:<10.4f} | {elapsed:<7.2f}\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

        if idx % 15 == 0 or idx == len(tfidf_grid):
            print(f"  Processed {idx}/{len(tfidf_grid)} TF-IDF runs... (Current Best Val F1: {max(r['val_f1_macro'] for r in tfidf_results):.4f})")

    # Find best TF-IDF
    best_tfidf = max(tfidf_results, key=lambda x: x["val_f1_macro"])
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(tfidf_table_sep)
        f.write(f"\n🏆 BEST TF-IDF CONFIGURATION (Selected via Validation Macro F1):\n")
        f.write(f"   Parameters     : {best_tfidf['params']}\n")
        f.write(f"   Validation Acc : {best_tfidf['val_acc']*100:.2f}%\n")
        f.write(f"   Validation F1  : {best_tfidf['val_f1_macro']:.4f}\n")
        f.write(f"   Golden Set Acc : {best_tfidf['gold_acc']*100:.2f}%\n")
        f.write(f"   Golden Set F1  : {best_tfidf['gold_f1_macro']:.4f}\n\n")

    # 5. GRID SEARCH FOR MODEL 2: DENSE SENTENCE TRANSFORMERS (all-MiniLM-L6-v2) + HEAD
    print("\n" + "=" * 90)
    print("SECTION B: GRID SEARCH ON DENSE EMBEDDINGS (all-MiniLM-L6-v2) + HEAD")
    print("=" * 90)
    print("Pre-computing dense embeddings once (takes ~30s)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    t_emb = time.time()
    X_train_emb = embedder.encode(X_train, show_progress_bar=False, batch_size=128)
    X_val_emb = embedder.encode(X_val, show_progress_bar=False, batch_size=128)
    X_gold_emb = embedder.encode(X_gold, show_progress_bar=False, batch_size=128)
    print(f"  -> Finished dense embeddings pre-computation in {time.time()-t_emb:.1f}s!")

    dense_grid = []
    for c_val in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        for solver in ["lbfgs", "saga"]:
            for cw in [None, "balanced"]:
                dense_grid.append({
                    "C": c_val,
                    "solver": solver,
                    "class_weight": cw
                })

    print(f"Total Dense Classifier Parameter Combinations to Evaluate: {len(dense_grid)}")

    dense_results = []
    dense_table_header = (
        f"{'Run':<5} | {'C':<6} | {'Solver':<8} | {'ClassWeight':<12} | "
        f"{'Val Acc':<8} | {'Val F1(M)':<9} | {'Gold Acc':<8} | {'Gold F1(M)':<10} | {'Time(s)':<7}\n"
    )
    dense_table_sep = "-" * 105 + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 130 + "\n")
        f.write("### SECTION B: DENSE (MiniLM-L6-v2) + LOGISTIC REGRESSION HEAD HYPERPARAMETER RESULTS\n")
        f.write(dense_table_header)
        f.write(dense_table_sep)

    for idx, params in enumerate(dense_grid, 1):
        t0 = time.time()
        clf = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            class_weight=params["class_weight"],
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train_emb, y_train)

        # Predict on ALL 7,496 Val
        val_pred = clf.predict(X_val_emb)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_f1_weighted = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        # Predict on 200 Golden Set
        gold_pred = clf.predict(X_gold_emb)
        gold_acc = accuracy_score(y_gold, gold_pred)
        gold_f1_macro = f1_score(y_gold, gold_pred, average="macro", zero_division=0)

        elapsed = time.time() - t0

        record = {
            "run": idx,
            "params": params,
            "val_acc": val_acc,
            "val_f1_macro": val_f1_macro,
            "val_f1_weighted": val_f1_weighted,
            "gold_acc": gold_acc,
            "gold_f1_macro": gold_f1_macro,
            "elapsed": elapsed
        }
        dense_results.append(record)

        log_line = (
            f"D{idx:02d}   | {params['C']:<6.2f} | {params['solver']:<8} | "
            f"{str(params['class_weight']):<12} | "
            f"{val_acc*100:<7.2f}% | {val_f1_macro:<9.4f} | {gold_acc*100:<7.2f}% | {gold_f1_macro:<10.4f} | {elapsed:<7.2f}\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)

        if idx % 10 == 0 or idx == len(dense_grid):
            print(f"  Processed {idx}/{len(dense_grid)} Dense runs... (Current Best Val F1: {max(r['val_f1_macro'] for r in dense_results):.4f})")

    # Find best Dense model
    best_dense = max(dense_results, key=lambda x: x["val_f1_macro"])
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(dense_table_sep)
        f.write(f"\n🏆 BEST DENSE EMBEDDING CONFIGURATION (Selected via Validation Macro F1):\n")
        f.write(f"   Parameters     : {best_dense['params']}\n")
        f.write(f"   Validation Acc : {best_dense['val_acc']*100:.2f}%\n")
        f.write(f"   Validation F1  : {best_dense['val_f1_macro']:.4f}\n")
        f.write(f"   Golden Set Acc : {best_dense['gold_acc']*100:.2f}%\n")
        f.write(f"   Golden Set F1  : {best_dense['gold_f1_macro']:.4f}\n")
        f.write("=" * 130 + "\n")

    total_time = time.time() - start_total
    print("\n" + "=" * 90)
    print(f"ALL TUNING EXPERIMENTS COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} mins)!")
    print(f"Detailed logs and tables written to: {log_file}")
    print("=" * 90)

if __name__ == "__main__":
    run_exhaustive_hyperparam_tuning()
