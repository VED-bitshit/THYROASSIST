import streamlit as st
import pandas as pd
import time

from agent1_risk_scoring import RiskScoringAgent
from agent2_knowledge_retriever import retrieve_evidence
from agent3_reasoning import ReasoningAgent
from agent4_summary import SummaryAgent


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Agentic AI – Thyroid Risk Triage",
    layout="wide"
)

# =========================================================
# DECORATIVE CSS (SAFE)
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}

.block {
    background-color: #020617;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 18px rgba(59,130,246,0.18);
}

.risk-low { color: #22c55e; font-weight: 800; }
.risk-moderate { color: #facc15; font-weight: 800; }
.risk-high { color: #ef4444; font-weight: 800; }

.footer {
    font-size: 13px;
    color: #9ca3af;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>Agentic AI – Thyroid Risk Triage</h1>",
    unsafe_allow_html=True
)
st.caption("Clinical Decision Support System (Educational Use Only)")

# =========================================================
# LOAD MODEL ONCE
# =========================================================
@st.cache_resource
def load_agent():
    agent = RiskScoringAgent()
    agent.load_model("model.pkl")
    return agent

risk_agent = load_agent()

# =========================================================
# FORM
# =========================================================
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Details")
        age = st.number_input("Age", min_value=1, max_value=120)
        sex = st.selectbox("Sex", ["m", "f"])

        st.subheader("Clinical History")
        goitre = st.checkbox("Goitre")
        tumor = st.checkbox("Tumor")
        pregnant = st.checkbox("Pregnant")
        thyroid_surgery = st.checkbox("History of Thyroid Surgery")
        on_thyroxine = st.checkbox("On Thyroxine")
        on_antithyroid_medication = st.checkbox("On Anti-thyroid Medication")

    with col2:
        st.subheader("Additional Factors")
        lithium = st.checkbox("Lithium Usage")
        sick = st.checkbox("Currently Sick")

        st.subheader("Laboratory Findings")
        TSH_measured = st.checkbox("TSH Measured")
        TSH = st.checkbox("TSH Abnormal") if TSH_measured else False

        T3_measured = st.checkbox("T3 Measured")
        T3 = st.checkbox("T3 Abnormal") if T3_measured else False

        TT4_measured = st.checkbox("TT4 Measured")
        TT4 = st.checkbox("TT4 Abnormal") if TT4_measured else False

        FTI_measured = st.checkbox("FTI Measured")
        FTI = st.checkbox("FTI Abnormal") if FTI_measured else False

    submit = st.form_submit_button("Run Risk Assessment")

# =========================================================
# PIPELINE
# =========================================================
if submit:

    patient_data = {
        "age": age,
        "sex": sex,
        "goitre": goitre,
        "tumor": tumor,
        "pregnant": pregnant,
        "thyroid_surgery": thyroid_surgery,
        "on_thyroxine": on_thyroxine,
        "on_antithyroid_medication": on_antithyroid_medication,
        "lithium": lithium,
        "sick": sick,
        "TSH_measured": TSH_measured,
        "TSH": TSH,
        "T3_measured": T3_measured,
        "T3": T3,
        "TT4_measured": TT4_measured,
        "TT4": TT4,
        "FTI_measured": FTI_measured,
        "FTI": FTI,
    }

    progress = st.progress(0)

    # ---------------- Agent 1 ----------------
    with st.spinner("Agent 1: Risk Prediction"):
        time.sleep(0.4)
        patient_df = pd.DataFrame([patient_data])
        risk_obj = risk_agent.predict(patient_df)
        progress.progress(30)

    risk_prediction = {
        "risk_level": risk_obj.risk_level.name,
        "risk_score": risk_obj.risk_score,
        "uncertainty_flags": risk_obj.uncertainty_flags
    }

    # ---------------- Agent 2 (SAFE) ----------------
    with st.spinner("Agent 2: Knowledge Retrieval"):
        time.sleep(0.4)
        try:
            evidence = retrieve_evidence(risk_prediction)
            if evidence is None:
                evidence = []
        except Exception:
            evidence = []
        progress.progress(55)

    # ---------------- Agent 3 ----------------
    with st.spinner("Agent 3: Clinical Reasoning"):
        time.sleep(0.4)
        reasoning_agent = ReasoningAgent()
        reasoning = reasoning_agent.generate_reasoning(
            risk_prediction, patient_data, evidence
        )
        progress.progress(80)

    # ---------------- Agent 4 ----------------
    with st.spinner("Agent 4: Summary Generation"):
        time.sleep(0.4)
        summary_agent = SummaryAgent()
        doctor_summary, patient_summary = summary_agent.generate_summary(reasoning)
        progress.progress(100)

    st.success("Assessment Completed Successfully")

    # =========================================================
    # OUTPUT
    # =========================================================
    risk_class = {
        "LOW": "risk-low",
        "MODERATE": "risk-moderate",
        "HIGH": "risk-high"
    }.get(risk_prediction["risk_level"], "risk-low")

    st.markdown(
        f"<div class='block'>"
        f"<h3>Risk Level: <span class='{risk_class}'>{risk_prediction['risk_level']}</span></h3>"
        f"<p>Risk Score: {risk_prediction['risk_score']:.2f}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Doctor Summary", "Patient Summary"])

    with tab1:
        st.text_area("Clinical Interpretation", doctor_summary, height=220)

    with tab2:
        st.text_area("Patient Explanation", patient_summary, height=200)

    # =========================================================
    # DOWNLOAD REPORT
    # =========================================================
    report_text = f"""
THYROID RISK TRIAGE REPORT

Risk Level: {risk_prediction['risk_level']}
Risk Score: {risk_prediction['risk_score']:.2f}

--- Clinical Reasoning ---
{chr(10).join(reasoning)}

--- Doctor Summary ---
{doctor_summary}

--- Patient Summary ---
{patient_summary}

DISCLAIMER:
This system provides clinical decision support only.
"""

    st.download_button(
        label="Download Report (TXT)",
        data=report_text,
        file_name="thyroid_risk_report.txt",
        mime="text/plain"
    )

    st.markdown(
        "<div class='footer'>This system is not a substitute for professional medical advice.</div>",
        unsafe_allow_html=True
    )