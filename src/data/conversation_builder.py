import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any

def clean_text(text: str) -> str:
    """Basic text cleanup: strip extra whitespace and normalize encoding."""
    if not text:
        return ""
    # Normalize unicode / whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_amazon_conversations(
    raw_csv_path: str = "data/raw/twcs/twcs.csv",
    output_jsonl_path: str = "data/processed/amazon_conversations.jsonl",
    max_conversations: int = 50000
):
    print(f"Reading raw tweets from {raw_csv_path} for AmazonHelp...")
    
    # Store tweets by id for Amazon interactions
    # 1. Identify all tweets where author_id == 'AmazonHelp' or text contains '@AmazonHelp'
    amazon_tweet_ids = set()
    tweets_by_id: Dict[str, Dict[str, Any]] = {}
    
    total_scanned = 0
    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_scanned += 1
            author = row["author_id"]
            text = row["text"]
            t_id = row["tweet_id"]
            
            is_amazon_brand = (author == "AmazonHelp")
            mentions_amazon = ("@AmazonHelp" in text or "@amazonhelp" in text)
            
            if is_amazon_brand or mentions_amazon:
                amazon_tweet_ids.add(t_id)
                tweets_by_id[t_id] = {
                    "tweet_id": t_id,
                    "author_id": author,
                    "inbound": row["inbound"] == "True",
                    "created_at": row["created_at"],
                    "text": clean_text(text),
                    "response_tweet_id": row["response_tweet_id"].strip() if row["response_tweet_id"] else None,
                    "in_response_to_tweet_id": row["in_response_to_tweet_id"].strip().replace(".0", "") if row["in_response_to_tweet_id"] else None
                }
            
            if total_scanned % 500000 == 0:
                print(f"  Scanned {total_scanned:,} rows... Captured {len(tweets_by_id):,} Amazon tweets.")

    print(f"\nTotal Amazon-related tweets loaded: {len(tweets_by_id):,}")
    print("Reconstructing conversations (Customer Issue -> Brand Response)...")

    conversations = []
    seen_customer_texts = set()

    for t_id, tweet in tweets_by_id.items():
        # Look for customer tweets directed at Amazon
        if tweet["inbound"] and tweet["response_tweet_id"]:
            cust_text = tweet["text"]
            
            # Simple quality filter on customer issue
            # Must be at least 15 characters and not just an @mention
            words = cust_text.split()
            if len(words) < 4 or len(cust_text) < 15:
                continue
            
            # Avoid exact duplicate customer texts
            norm_cust_text = cust_text.lower()
            if norm_cust_text in seen_customer_texts:
                continue
            
            # Check responses for AmazonHelp reply
            resp_ids = [r.strip() for r in tweet["response_tweet_id"].split(",") if r.strip()]
            for r_id in resp_ids:
                if r_id in tweets_by_id:
                    resp_tweet = tweets_by_id[r_id]
                    if resp_tweet["author_id"] == "AmazonHelp":
                        brand_text = resp_tweet["text"]
                        
                        # Filter out empty or trivial replies
                        if len(brand_text) > 15:
                            seen_customer_texts.add(norm_cust_text)
                            conversations.append({
                                "conversation_id": f"conv_{tweet['tweet_id']}",
                                "customer_tweet_id": tweet["tweet_id"],
                                "customer_author_id": tweet["author_id"],
                                "customer_text": cust_text,
                                "customer_created_at": tweet["created_at"],
                                "brand_tweet_id": resp_tweet["tweet_id"],
                                "brand_text": brand_text,
                                "brand_created_at": resp_tweet["created_at"]
                            })
                            break
        
        if len(conversations) >= max_conversations:
            break

    print(f"\nSuccessfully extracted {len(conversations):,} high-quality Customer -> Brand conversations!")
    
    # Save to JSONL
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    print(f"Saved to: {output_jsonl_path}")
    
    # Print 3 examples
    print("\n" + "="*60)
    print("SAMPLE CONVERSATIONS EXTRACTED:")
    print("="*60)
    for i, sample in enumerate(conversations[:3], 1):
        print(f"\n--- Conversation #{i} (ID: {sample['conversation_id']}) ---")
        print(f"Customer: {sample['customer_text']}")
        print(f"Amazon:   {sample['brand_text']}")

if __name__ == "__main__":
    build_amazon_conversations()
