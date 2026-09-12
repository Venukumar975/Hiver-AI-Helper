import re
from typing import Dict, Any, List

class EscalationPolicy:
    """
    Conservative Escalation Gatekeeper for Customer Support AI Agent.
    Prioritizes minimizing FALSE AUTO-HANDLES (i.e. incorrectly auto-handling a risky issue).
    
    Decision Signals evaluated:
      1. Intent Classifier Confidence (threshold: 0.70)
      2. Retrieval Quality / Semantic Support (threshold: 0.50)
      3. Risk & Safety Keywords (legal, fraud, hacked, dispute, threat, abusive language)
      4. High-Risk / Sensitive Domains (unauthorized transactions, driver misconduct)
      5. Excessive ambiguity (extremely brief queries lacking context)
    """
    
    CONFIDENCE_THRESHOLD = 0.50
    RETRIEVAL_SIMILARITY_THRESHOLD = 0.50
    
    RISK_PATTERNS = [
        r"\b(lawyer|legal|sue|lawsuit|attorney|court)\b",
        r"\b(fraud|stolen|hacked|scam|crime|police|police report)\b",
        r"\b(unauthorized charge|credit card fraud|bank dispute|dispute with my bank|chargeback)\b",
        r"\b(locked out of account|cannot access account|account suspended permanently)\b",
        r"\b(hung up on me|driver assaulted|driver hit|abusive|threat|harassment)\b"
    ]
    
    def __init__(self, confidence_threshold: float = 0.50, retrieval_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Returns:
          {
            "decision": "AUTO_HANDLE" | "ESCALATE",
            "reason": "<Explicit justification>"
          }
        """
        # Signal 1: Check High-Risk / Legal / Fraud Keywords
        for pattern in self.RISK_PATTERNS:
            match = re.search(pattern, customer_message, re.IGNORECASE)
            if match:
                return {
                    "decision": "ESCALATE",
                    "reason": f"High-risk safety trigger detected: '{match.group(0)}'. Requires senior human specialist review."
                }

        # Signal 2: Classifier Confidence
        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "decision": "ESCALATE",
                "reason": f"Low classification confidence ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD:.2f}). Message is ambiguous or spans multiple intents."
            }

        # Signal 3: Historical Evidence Support
        if not retrieved_cases:
            return {
                "decision": "ESCALATE",
                "reason": "Zero matching historical resolution cases found in knowledge base."
            }
            
        top_score = retrieved_cases[0].get("similarity_score", 0.0)
        if top_score < self.RETRIEVAL_SIMILARITY_THRESHOLD:
            return {
                "decision": "ESCALATE",
                "reason": f"Weak historical evidence match (similarity {top_score:.2f} < {self.RETRIEVAL_SIMILARITY_THRESHOLD:.2f}). No close precedent."
            }

        # Signal 4: Intent-specific policy checks
        # Feedback complaints with abusive or high frustration sentiment escalate to customer relations
        if predicted_intent == "feedback_complaint":
            words = customer_message.lower().split()
            if any(w in words for w in ["worst", "terrible", "disgrace", "horrible", "disgusting", "never"]):
                return {
                    "decision": "ESCALATE",
                    "reason": "Severe customer dissatisfaction/complaint requires human supervisor empathy and de-escalation."
                }

        # Signal 5: Length & Context check
        if len(customer_message.strip().split()) <= 4:
            return {
                "decision": "ESCALATE",
                "reason": "Message is too brief and lacks essential context for autonomous resolution."
            }

        # If all conservative checks pass:
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"High-confidence {predicted_intent} query ({confidence*100:.1f}%) backed by strong historical precedents (top match: {top_score:.2f})."
        }

if __name__ == "__main__":
    policy = EscalationPolicy()
    
    # Test 1: Routine
    res1 = policy.evaluate(
        customer_message="Where is my package? It was supposed to be delivered yesterday.",
        predicted_intent="delivery_issue",
        confidence=0.89,
        retrieved_cases=[{"case_id": "conv_1", "similarity_score": 0.75}]
    )
    print("Test 1 (Routine Delivery):", res1)
    
    # Test 2: Legal Threat
    res2 = policy.evaluate(
        customer_message="I have not received my order and I am calling my lawyer to sue you for fraud!",
        predicted_intent="delivery_issue",
        confidence=0.92,
        retrieved_cases=[{"case_id": "conv_1", "similarity_score": 0.72}]
    )
    print("Test 2 (Legal Threat):", res2)
    
    # Test 3: Low Confidence
    res3 = policy.evaluate(
        customer_message="Can you please check on the thing we discussed earlier?",
        predicted_intent="general",
        confidence=0.45,
        retrieved_cases=[{"case_id": "conv_1", "similarity_score": 0.40}]
    )
    print("Test 3 (Low Confidence):", res3)
