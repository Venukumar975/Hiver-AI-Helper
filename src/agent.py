from typing import Dict, Any, List
from src.classification.classifier import IntentClassifier
from src.retrieval.retriever import HistoricalCaseRetriever
from src.generation.generator import ResponseGenerator
from src.escalation.policy import EscalationPolicy

class SupportAgent:
    """
    Unified End-to-End AI Customer Support Agent for AmazonHelp.
    
    Complete Pipeline:
      1. Clean & Validate Input
      2. Classify Intent & Compute Calibrated Confidence
      3. Retrieve Intent-Filtered Top-K Historical Cases via FAISS
      4. Evaluate Conservative Escalation Gatekeeper
      5. Generate Grounded Draft Response citing Evidence Cases
    """
    def __init__(self):
        print("Initializing SupportAgent components...")
        self.classifier = IntentClassifier()
        self.classifier.load()
        self.retriever = HistoricalCaseRetriever()
        self.escalation_policy = EscalationPolicy()
        self.generator = ResponseGenerator()
        print("SupportAgent fully initialized and ready!")

    def handle(self, message: str) -> Dict[str, Any]:
        """
        Single entrypoint for customer support automation.
        """
        # 1. Input Validation
        cleaned_msg = message.strip()
        if not cleaned_msg:
            return {
                "customer_message": message,
                "intent": "unknown",
                "confidence": 0.0,
                "decision": "ESCALATE",
                "decision_reason": "Empty message received.",
                "draft_response": "Hello! How can we assist you today? Please provide more details regarding your inquiry. ^AH",
                "evidence_case_ids": [],
                "retrieved_cases": []
            }

        # 2. Intent Classification + Confidence
        clf_result = self.classifier.predict(cleaned_msg)
        predicted_intent = clf_result["intent"]
        confidence = clf_result["confidence"]

        # 3. Intent-Aware Retrieval from FAISS
        retrieved_cases = self.retriever.retrieve_intent_aware(
            query=cleaned_msg,
            intent=predicted_intent,
            top_k=3
        )

        # 4. Conservative Escalation Policy Decision
        esc_result = self.escalation_policy.evaluate(
            customer_message=cleaned_msg,
            predicted_intent=predicted_intent,
            confidence=confidence,
            retrieved_cases=retrieved_cases
        )
        decision = esc_result["decision"]
        decision_reason = esc_result["reason"]

        # 5. Grounded Response Generation
        gen_result = self.generator.generate_grounded_response(
            customer_message=cleaned_msg,
            predicted_intent=predicted_intent,
            retrieved_cases=retrieved_cases,
            decision=decision,
            decision_reason=decision_reason
        )

        return {
            "customer_message": cleaned_msg,
            "intent": predicted_intent,
            "confidence": confidence,
            "decision": decision,
            "decision_reason": decision_reason,
            "draft_response": gen_result["draft_response"],
            "evidence_case_ids": gen_result["evidence_case_ids"],
            "retrieved_cases": retrieved_cases
        }

if __name__ == "__main__":
    agent = SupportAgent()
    
    print("\n" + "="*70)
    print("TEST 1: ROUTINE QUERY (EXPECT AUTO_HANDLE)")
    print("="*70)
    res1 = agent.handle("My package was supposed to arrive today by 8 PM but still hasn't arrived. Tracking ID 93810481.")
    print(f"Intent    : {res1['intent']} (Confidence: {res1['confidence']:.2f})")
    print(f"Decision  : {res1['decision']}")
    print(f"Reason    : {res1['decision_reason']}")
    print(f"Draft     : {res1['draft_response']}")
    print(f"Evidence  : {res1['evidence_case_ids']}")

    print("\n" + "="*70)
    print("TEST 2: HIGH-RISK / LEGAL THREAT (EXPECT ESCALATE)")
    print("="*70)
    res2 = agent.handle("Someone hacked into my account, stole my money, and I am filing a police report and calling a lawyer!")
    print(f"Intent    : {res2['intent']} (Confidence: {res2['confidence']:.2f})")
    print(f"Decision  : {res2['decision']}")
    print(f"Reason    : {res2['decision_reason']}")
    print(f"Draft     : {res2['draft_response']}")
    print(f"Evidence  : {res2['evidence_case_ids']}")
