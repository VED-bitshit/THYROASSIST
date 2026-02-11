# agent3_reasoning.py
# FINAL POLYMORPHIC VERSION – CLI + UI SAFE

class ReasoningAgent:
    """
    Reasoning & Explainability Agent
    Works with:
    - RiskPrediction object (CLI)
    - dict (Streamlit UI)
    """

    def __init__(self):
        pass

    def generate_reasoning(self, risk_prediction, patient_data, evidence):
        reasoning = []

        # ---------------- SAFE ACCESS ----------------
        if isinstance(risk_prediction, dict):
            risk_level = risk_prediction.get("risk_level", "UNKNOWN")
            risk_score = risk_prediction.get("risk_score", 0.0)
            uncertainty_flags = risk_prediction.get("uncertainty_flags", [])
        else:
            # RiskPrediction object
            risk_level = risk_prediction.risk_level.name
            risk_score = risk_prediction.risk_score
            uncertainty_flags = risk_prediction.uncertainty_flags

        # ---------------- REASONING ----------------
        reasoning.append(
            f"Predicted thyroid risk level: {risk_level} "
            f"(risk score: {risk_score:.2f})"
        )

        # Feature-based reasoning
        if patient_data.get("TSH") is True:
            reasoning.append("Abnormal TSH level detected")

        if patient_data.get("thyroid_surgery"):
            reasoning.append("History of thyroid surgery increases monitoring needs")

        if patient_data.get("on_antithyroid_medication"):
            reasoning.append("Patient is on anti-thyroid medication")

        # Evidence
        if evidence:
            reasoning.append("Relevant clinical guidance:")
            for e in evidence:
                reasoning.append(f"- {e}")

        # Uncertainty
        if uncertainty_flags:
            reasoning.append("Uncertainty factors identified:")
            for flag in uncertainty_flags:
                reasoning.append(f"- {flag}")

        # Safety disclaimer
        reasoning.append(
            "This assessment is for clinical decision support only "
            "and is not a medical diagnosis."
        )

        return reasoning


# ---------------------------------------------------------
# BACKWARD COMPATIBILITY (DO NOT REMOVE)
# ---------------------------------------------------------

class ClinicalReasoning(ReasoningAgent):
    pass
