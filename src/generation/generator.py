import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class ResponseGenerator:
    """
    Grounded Response Generator powered by Google Gemini API.
    
    Responsibilities:
      1. Draft an empathetic, brand-aligned reply mimicking Amazon Twitter support persona.
      2. Ground the advice strictly in retrieved historical resolutions.
      3. Never hallucinate policies or invent promises.
      4. Explicitly cite evidence case IDs.
    """
    ACTIVE_MODELS = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash"
    ]
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Notice initializing Gemini client: {e}")

    def generate_grounded_response(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]],
        decision: str = "AUTO_HANDLE",
        decision_reason: str = ""
    ) -> Dict[str, Any]:
        evidence_ids = [c["case_id"] for c in retrieved_cases]
        
        # In ESCALATE mode, form a polite human-handoff reply
        if decision == "ESCALATE":
            draft = (
                "I understand how important this issue is, and I want to ensure you receive the specialized "
                "assistance you deserve. I have routed your case directly to a senior customer support specialist "
                f"for immediate review. Please stay tuned while our human team investigates this for you."
            )
            return {
                "draft_response": draft,
                "reasoning_summary": f"Case escalated to human specialist. Reason: {decision_reason}",
                "evidence_case_ids": evidence_ids
            }

        # Build context from top-3 retrieved historical cases
        precedents_text = ""
        for i, c in enumerate(retrieved_cases[:3], 1):
            precedents_text += f"\n[Historical Precedent #{i} | Case ID: {c['case_id']}]\n"
            precedents_text += f"Customer: {c['customer_issue']}\n"
            precedents_text += f"Amazon Resolution: {c['brand_resolution']}\n"

        # Attempt Gemini Generation
        if self.client:
            prompt = f"""You are an expert customer service AI agent for Amazon customer support on Twitter (@AmazonHelp).
Your tone must be polite, concise, professional, and empathetic (typically ending with signoff ^AH).
GROUNDEDNESS REQUIREMENT: Base your answer strictly on the historical support precedents provided below. Do NOT invent policies or make unauthorized promises.

Customer Tweet: "{customer_message}"
Predicted Intent: {predicted_intent}

Historical Support Precedents:
{precedents_text}

Draft a 2-sentence grounded reply for Twitter following the guidance in the precedents:"""
            
            for m in self.ACTIVE_MODELS:
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    draft_text = response.text.strip()
                    # Strip quotes if enclosed
                    if draft_text.startswith('"') and draft_text.endswith('"'):
                        draft_text = draft_text[1:-1]
                    return {
                        "draft_response": draft_text,
                        "reasoning_summary": f"Generated via {m}, grounded in {len(retrieved_cases[:3])} historical precedents.",
                        "evidence_case_ids": evidence_ids
                    }
                except Exception as e:
                    continue

        # Grounded Fallback Synthesizer (if offline or API rate-limited)
        primary = retrieved_cases[0] if retrieved_cases else None
        if primary:
            clean_res = re.sub(r"@\w+", "", primary["brand_resolution"])
            clean_res = re.sub(r"\^[A-Z]{2}", "", clean_res).strip()
            draft = f"We're sorry to hear about this issue! {clean_res} ^AH"
        else:
            draft = "Thank you for reaching out to Amazon Help. Please contact us via secure chat so we can look into your order. ^AH"

        return {
            "draft_response": draft,
            "reasoning_summary": f"Grounded response drafted using historical precedent {primary['case_id'] if primary else 'knowledge base'}.",
            "evidence_case_ids": evidence_ids
        }

if __name__ == "__main__":
    gen = ResponseGenerator()
    dummy_cases = [
        {
            "case_id": "conv_707391",
            "customer_issue": "Package marked delivered on Amazon, but it has not arrived in my mailbox. What do I do?",
            "brand_resolution": "Here are some helpful tips on finding your package: check with household/neighbors. Keep us updated! ^HM"
        }
    ]
    res = gen.generate_grounded_response(
        customer_message="My package says delivered today but I checked my porch and nothing is there!",
        predicted_intent="delivery_issue",
        retrieved_cases=dummy_cases
    )
    print("Draft Reply via Gemini:\n", res["draft_response"])
    print("\nReasoning:\n", res["reasoning_summary"])
    print("\nEvidence Case IDs:\n", res["evidence_case_ids"])
