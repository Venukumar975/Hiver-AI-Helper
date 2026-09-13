import re
from typing import Dict, Any, List, Optional

class EscalationPolicy:
    """
    Conservative Escalation Gatekeeper for Customer Support AI Agent.
    Prioritizes minimizing FALSE AUTO-HANDLES (i.e. incorrectly auto-handling a risky issue).
    
    Two-Stage Architecture:
      Stage 1 (Pre-Retrieval Gate):
        - Evaluates Intent Classifier Confidence (threshold: 0.50, 0.25 for greetings)
        - Risk & Safety Keywords (legal, fraud, hacked, dispute, threat, abusive language)
        - High-frustration sentiment on complaints
        - Message brevity / context absence
        -> If ANY check fails: Immediately ESCALATE (Skip RAG, 0 docs retrieved).
      Stage 2 (Post-Retrieval Gate):
        - Evaluates historical precedent similarity (threshold: 0.50)
        -> If top match < 0.50: ESCALATE (weak evidence).
        -> If match >= 0.50: AUTO_HANDLE with grounded draft.
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

    def evaluate_pre_retrieval(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float
    ) -> Dict[str, str]:
        """
        Stage 1: Pre-Retrieval Safety & Confidence Gate.
        Decides whether it is even safe / necessary to query the FAISS RAG index.
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
        effective_conf_thresh = 0.25 if predicted_intent == "general_inquiry_greeting" else self.CONFIDENCE_THRESHOLD
        if confidence < effective_conf_thresh:
            return {
                "decision": "ESCALATE",
                "reason": f"Low classification confidence ({confidence:.2f} < {effective_conf_thresh:.2f}). Message is ambiguous or spans multiple intents."
            }

        # Signal 3: Intent-specific policy checks
        if predicted_intent == "feedback_complaint":
            words = customer_message.lower().split()
            if any(w in words for w in ["worst", "terrible", "disgrace", "horrible", "disgusting", "never"]):
                return {
                    "decision": "ESCALATE",
                    "reason": "Severe customer dissatisfaction/complaint requires human supervisor empathy and de-escalation."
                }

        # Signal 4: Length & Context check (exempt greetings/pleasantries as they are naturally brief)
        if predicted_intent != "general_inquiry_greeting" and len(customer_message.strip().split()) <= 4:
            return {
                "decision": "ESCALATE",
                "reason": "Message is too brief and lacks essential context for autonomous resolution."
            }

        # Passes pre-retrieval checks -> Safe to query RAG
        return {
            "decision": "PROCEED_TO_RETRIEVAL",
            "reason": f"High-confidence {predicted_intent} ({confidence*100:.1f}%). Safe to retrieve precedents."
        }

    def evaluate_retrieval_quality(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        retrieved_cases: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Stage 2: Post-Retrieval Quality Gate.
        Verifies that retrieved precedents are close enough to safely ground a response.
        """
        if not retrieved_cases:
            return {
                "decision": "ESCALATE",
                "reason": "Zero matching historical resolution cases found in knowledge base."
            }
            
        top_score = retrieved_cases[0].get("similarity_score", 0.0)
        effective_ret_thresh = 0.30 if predicted_intent == "general_inquiry_greeting" else self.RETRIEVAL_SIMILARITY_THRESHOLD
        if top_score < effective_ret_thresh:
            return {
                "decision": "ESCALATE",
                "reason": f"Weak historical evidence match (similarity {top_score:.2f} < {effective_ret_thresh:.2f}). No close precedent."
            }

        return {
            "decision": "AUTO_HANDLE",
            "reason": f"High-confidence {predicted_intent} query ({confidence*100:.1f}%) backed by strong historical precedents (top match: {top_score:.2f})."
        }

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        confidence: float,
        retrieved_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Unified evaluation endpoint (combines Stage 1 and Stage 2).
        """
        pre_result = self.evaluate_pre_retrieval(customer_message, predicted_intent, confidence)
        if pre_result["decision"] == "ESCALATE":
            return pre_result

        return self.evaluate_retrieval_quality(
            customer_message,
            predicted_intent,
            confidence,
            retrieved_cases or []
        )

if __name__ == "__main__":
    policy = EscalationPolicy()
    
    # Test 1: Low confidence (< 0.50) -> Should ESCALATE before retrieval
    p1 = policy.evaluate_pre_retrieval("I want to know", "delivery_issue", 0.38)
    print("Test 1 (Low Conf):", p1)

    # Test 2: High confidence (0.85) -> Should PROCEED_TO_RETRIEVAL
    p2 = policy.evaluate_pre_retrieval("Where is my package? Tracking 123", "delivery_issue", 0.85)
    print("Test 2 (High Conf):", p2)
