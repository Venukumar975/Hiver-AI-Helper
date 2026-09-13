import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.model_loader import load_models_from_ini

load_dotenv()

class ResponseGenerator:
    """
    Grounded Response Generator powered by Google Gemini API with prioritized fallback.
    
    Responsibilities:
      1. Loads models from models.ini sorted by priority (high reasoning + tokens).
      2. Tries the highest priority model first; if unavailable, exhausted (429), or busy (503),
         automatically falls back to the next model in sequence.
      3. Drafts an empathetic, brand-aligned reply mimicking Amazon Twitter support persona (^AH).
      4. Grounds advice strictly in retrieved historical resolutions without hallucination.
      5. Explicitly cites evidence case IDs and reports the exact model used and fallback chain.
      6. Offline fallback synthesizer ensures 100% uptime even during total API outage.
    """
    def __init__(self, api_key: Optional[str] = None, models_ini_path: str = "models.ini"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.models_ini_path = models_ini_path
        self.client = None
        self.active_models: List[str] = load_models_from_ini(models_ini_path)
        self.blocked_models: set = set()

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Generator Init] Notice initializing Gemini client: {e}")

    def block_model(self, model_name: str) -> None:
        """Dynamically block a model to test or simulate quota exhaustion / outages."""
        self.blocked_models.add(model_name)

    def unblock_model(self, model_name: str) -> None:
        """Unblock a previously blocked model."""
        self.blocked_models.discard(model_name)

    def reset_blocked_models(self) -> None:
        """Clear all blocked models."""
        self.blocked_models.clear()

    def generate_grounded_response(
        self,
        customer_message: str,
        predicted_intent: str,
        retrieved_cases: List[Dict[str, Any]],
        decision: str = "AUTO_HANDLE",
        decision_reason: str = "",
        override_blocked_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        evidence_ids = [c["case_id"] for c in retrieved_cases]
        blocked = set(self.blocked_models)
        if override_blocked_models:
            blocked.update(override_blocked_models)
        
        # In ESCALATE mode, understand the issue, state specialist handoff, and ask for needed details (Order ID, etc.)
        if decision == "ESCALATE":
            # Context-specific clarification request based on issue type
            if predicted_intent in ["account_prime_billing", "cancellation_change_order"]:
                ask_detail = "Could you please confirm your Order ID or account email via private message so we have everything ready?"
            elif predicted_intent in ["delivery_issue", "damaged_defective_item", "return_refund_status"]:
                ask_detail = "To help us expedite, could you please provide your 17-digit Order ID (e.g. 123-1234567-1234567) or tracking number via DM?"
            elif predicted_intent == "product_tech_support":
                ask_detail = "Could you share the specific device model (e.g. Echo Show, Fire TV Stick) and any error code you are seeing?"
            else:
                ask_detail = "Please send us a direct message with your order details or account information so we can investigate right away."

            # Dynamic escalation reply with human specialist handoff
            if self.client:
                esc_prompt = f"""You are a customer support agent for @AmazonHelp on Twitter.
The customer's message requires escalation to a human specialist.
Reason: {decision_reason}
Predicted Intent: {predicted_intent}
Customer Tweet: "{customer_message}"

TASK: Write a polite, concise (max 2 sentences) public Twitter reply that:
1. Reassures them that their case is being escalated to a senior support specialist for immediate review.
2. Prompts them to DM necessary information (like Order ID, tracking ID, or account details) depending on their issue.
3. Ends with the standard agent signoff ^AH.
Do NOT promise unauthorized refunds or make policy guarantees. Keep it professional and empathetic:"""
                for priority_idx, m in enumerate(self.active_models, 1):
                    if m in blocked:
                        continue
                    try:
                        response = self.client.models.generate_content(model=m, contents=esc_prompt)
                        draft_text = response.text.strip().strip('"')
                        if not draft_text.endswith("^AH"):
                            draft_text += " ^AH"
                        return {
                            "draft_response": draft_text,
                            "reasoning_summary": f"Context-aware escalation handoff via Priority #{priority_idx} ({m}). Reason: {decision_reason}",
                            "evidence_case_ids": evidence_ids,
                            "model_used": m,
                            "priority_level": priority_idx,
                            "fallback_chain": [{"priority": priority_idx, "model": m, "status": "success", "error": None}]
                        }
                    except Exception:
                        continue

            # Offline deterministic fallback with dynamic context ask
            draft = (
                f"We want to ensure this is handled properly, so I have routed your case directly to a customer "
                f"support specialist for review. {ask_detail} ^AH"
            )
            return {
                "draft_response": draft,
                "reasoning_summary": f"Case escalated to human specialist with detail request. Reason: {decision_reason}",
                "evidence_case_ids": evidence_ids,
                "model_used": "escalation_policy_engine",
                "fallback_chain": []
            }

        # Build context from top-3 retrieved historical cases (cleaning trailing colons or broken fragments)
        precedents_text = ""
        for i, c in enumerate(retrieved_cases[:3], 1):
            clean_brand_res = re.sub(r":\s*$", ".", c['brand_resolution'].strip())
            clean_brand_res = re.sub(r":\s+They'll", ". They'll", clean_brand_res)
            clean_brand_res = re.sub(r":\s+Please", ". Please", clean_brand_res)
            clean_brand_res = re.sub(r":\s+We'll", ". We'll", clean_brand_res)
            precedents_text += f"\n[Historical Precedent #{i} | Case ID: {c['case_id']}]\n"
            precedents_text += f"Customer Issue: {c['customer_issue']}\n"
            precedents_text += f"Amazon Resolution: {clean_brand_res}\n"

        blocked = set(self.blocked_models)
        if override_blocked_models:
            blocked.update(override_blocked_models)

        fallback_chain: List[Dict[str, Any]] = []

        # Attempt Gemini Generation across models in priority order
        if self.client:
            prompt = f"""You are an Amazon Customer Support agent on Twitter (@AmazonHelp).
Your task is to draft a grounded 2-sentence response to the customer using ONLY the policies and actions shown in the historical precedents.

STRICT GROUNDING & ZERO-HALLUCINATION RULES:
1. Do NOT invent, guess, or output any URLs or web links (do NOT create fake 'amzn.to' links or placeholders). If the customer needs direct support, advise them to send a Direct Message (DM) or check their Amazon order history directly.
2. Rely strictly on the solutions shown in the Historical Support Precedents.
3. Tone must be polite, helpful, and empathetic.
4. Always end the response with the agent identifier ^AH.

Customer Tweet: "{customer_message}"
Predicted Intent: {predicted_intent}

Historical Support Precedents:
{precedents_text}

Draft Grounded Reply:"""
            
            for priority_idx, m in enumerate(self.active_models, 1):
                if m in blocked:
                    msg = f"Model '{m}' is blocked (simulated outage/exhaustion). Skipping to next priority."
                    print(f"  [Fallback Priority {priority_idx}] {msg}")
                    fallback_chain.append({"priority": priority_idx, "model": m, "status": "blocked", "error": "Blocked by policy/test"})
                    continue

                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    draft_text = response.text.strip()
                    if draft_text.startswith('"') and draft_text.endswith('"'):
                        draft_text = draft_text[1:-1]

                    # Strip any accidental hallucinated markdown or http links
                    draft_text = re.sub(r"\[.*?\]\(https?://\S+\)", "via DM", draft_text)
                    draft_text = re.sub(r"https?://\S+", "", draft_text)
                    draft_text = re.sub(r"\s+", " ", draft_text).strip()
                    if not draft_text.endswith("^AH"):
                        draft_text += " ^AH"

                    fallback_chain.append({"priority": priority_idx, "model": m, "status": "success", "error": None})
                    print(f"  [Model Success] Generated reply using Priority #{priority_idx} model '{m}'.")

                    return {
                        "draft_response": draft_text,
                        "reasoning_summary": f"Generated via Priority #{priority_idx} ({m}), grounded in {len(retrieved_cases[:3])} historical precedents.",
                        "evidence_case_ids": evidence_ids,
                        "model_used": m,
                        "priority_level": priority_idx,
                        "fallback_chain": fallback_chain
                    }
                except Exception as e:
                    err_str = str(e)
                    # Extract concise error
                    status_reason = "503_busy" if "503" in err_str else ("429_quota" if "429" in err_str else "exception")
                    print(f"  [Fallback Priority {priority_idx}] Model '{m}' failed ({status_reason}). Error: {err_str[:60]}... Trying next priority...")
                    fallback_chain.append({"priority": priority_idx, "model": m, "status": status_reason, "error": err_str[:120]})
                    continue

        # Grounded Fallback Synthesizer (if all models failed or offline)
        print("  [Fallback Engine] All API models unavailable or exhausted. Using deterministic historical synthesizer.")
        primary = retrieved_cases[0] if retrieved_cases else None
        if primary:
            clean_res = re.sub(r"@\w+", "", primary["brand_resolution"])
            clean_res = re.sub(r"\^[A-Z]{2}", "", clean_res).strip()
            draft = f"We're sorry to hear about this issue! {clean_res} ^AH"
        else:
            draft = "Thank you for reaching out to Amazon Help. Please contact us via secure chat so we can look into your order. ^AH"

        return {
            "draft_response": draft,
            "reasoning_summary": f"Grounded response drafted using historical precedent {primary['case_id'] if primary else 'knowledge base'} (Offline Fallback).",
            "evidence_case_ids": evidence_ids,
            "model_used": "offline_precedent_synthesizer",
            "priority_level": 999,
            "fallback_chain": fallback_chain
        }

if __name__ == "__main__":
    gen = ResponseGenerator()
    print("Loaded models from models.ini in priority order:")
    for idx, model in enumerate(gen.active_models, 1):
        print(f"  Priority {idx}: {model}")

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
    print("\nModel Used:", res.get("model_used"))
    print("Draft Reply:\n", res["draft_response"])
    print("Fallback Chain:\n", res["fallback_chain"])
