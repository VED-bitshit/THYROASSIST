"""
Agent 4: Summary Agent
Generates two distinct summaries: technical for doctors, plain language for patients
"""

from typing import Dict, List
from dataclasses import dataclass

from agent1_risk_scoring import RiskPrediction
from agent2_knowledge_retriever import RetrievedEvidence
from agent3_reasoning import ClinicalReasoning


@dataclass
class DoctorSummary:
    """Technical summary for healthcare providers"""
    executive_summary: str
    risk_assessment: str
    key_findings: List[str]
    recommendations: List[str]
    evidence_citations: str
    next_steps: str


@dataclass
class PatientSummary:
    """Plain language summary for patients"""
    overview: str
    what_this_means: str
    next_steps: str
    questions_to_ask: List[str]
    reassurance: str


class SummaryAgent:
    """
    Agent 4: Generates tailored summaries for different audiences
    Creates clinical summaries for doctors and accessible summaries for patients
    """
    
    def __init__(self):
        pass
    
    def generate_doctor_summary(
        self,
        risk_prediction: RiskPrediction,
        patient_data: Dict,
        reasoning: ClinicalReasoning,
        evidence: List[RetrievedEvidence]
    ) -> DoctorSummary:
        """
        Generate structured clinical summary for healthcare providers
        
        Args:
            risk_prediction: Risk assessment from Agent 1
            patient_data: Original patient data
            reasoning: Clinical reasoning from Agent 3
            evidence: Retrieved guidelines from Agent 2
            
        Returns:
            DoctorSummary object
        """
        # Executive Summary
        executive = self._create_executive_summary(risk_prediction, patient_data)
        
        # Risk Assessment
        risk_assessment = self._create_risk_assessment(risk_prediction)
        
        # Key Findings
        key_findings = self._extract_key_findings(patient_data, risk_prediction)
        
        # Recommendations
        recommendations = self._create_recommendations(
            risk_prediction.risk_level.value, evidence
        )
        
        # Evidence Citations
        citations = self._format_evidence_citations(evidence)
        
        # Next Steps
        next_steps = self._create_doctor_next_steps(
            risk_prediction, patient_data, reasoning
        )
        
        return DoctorSummary(
            executive_summary=executive,
            risk_assessment=risk_assessment,
            key_findings=key_findings,
            recommendations=recommendations,
            evidence_citations=citations,
            next_steps=next_steps
        )
    
    def generate_patient_summary(
        self,
        risk_prediction: RiskPrediction,
        patient_data: Dict,
        reasoning: ClinicalReasoning
    ) -> PatientSummary:
        """
        Generate plain language summary for patients
        
        Args:
            risk_prediction: Risk assessment from Agent 1
            patient_data: Original patient data
            reasoning: Clinical reasoning from Agent 3
            
        Returns:
            PatientSummary object
        """
        # Overview
        overview = self._create_patient_overview(risk_prediction)
        
        # What This Means
        meaning = self._explain_to_patient(risk_prediction, patient_data)
        
        # Next Steps
        next_steps = self._create_patient_next_steps(risk_prediction)
        
        # Questions to Ask
        questions = self._generate_patient_questions(risk_prediction, patient_data)
        
        # Reassurance
        reassurance = self._create_reassurance(risk_prediction)
        
        return PatientSummary(
            overview=overview,
            what_this_means=meaning,
            next_steps=next_steps,
            questions_to_ask=questions,
            reassurance=reassurance
        )
    
    # Doctor Summary Components
    
    def _create_executive_summary(self, prediction: RiskPrediction, data: Dict) -> str:
        """Create brief executive summary"""
        age = data.get('age', 'Unknown')
        sex = data.get('sex', 'Unknown')
        
        summary = (
            f"Patient: {age}yo {sex.upper() if sex != 'Unknown' else 'Unknown sex'} | "
            f"Risk Level: {prediction.risk_level.value.upper()} | "
            f"Risk Score: {prediction.risk_score:.3f} | "
            f"Model: {prediction.model_name}"
        )
        
        return summary
    
    def _create_risk_assessment(self, prediction: RiskPrediction) -> str:
        """Detailed risk assessment"""
        assessment = []
        
        assessment.append(
            f"Risk Classification: {prediction.risk_level.value} "
            f"(Score: {prediction.risk_score:.3f})"
        )
        assessment.append(
            f"Confidence Interval: [{prediction.confidence_lower:.3f}, {prediction.confidence_upper:.3f}]"
        )
        
        if prediction.uncertainty_flags:
            assessment.append("\nUncertainty Factors:")
            for flag in prediction.uncertainty_flags:
                assessment.append(f"  - {flag}")
        
        return "\n".join(assessment)
    
    def _extract_key_findings(self, data: Dict, prediction: RiskPrediction) -> List[str]:
        """Extract key clinical findings"""
        findings = []
        
        # Abnormal labs
        lab_tests = ['TSH', 'T3', 'TT4', 'FTI']
        for test in lab_tests:
            if data.get(test) in [1, 't', True]:
                findings.append(f"Abnormal {test} level")
        
        # Significant history
        significant = [
            ('thyroid_surgery', 'History of thyroid surgery'),
            ('on_antithyroid_medication', 'Currently on antithyroid medication'),
            ('on_thyroxine', 'Currently on thyroxine'),
            ('goitre', 'Goiter present'),
            ('tumor', 'Thyroid tumor'),
            ('pregnant', 'Currently pregnant')
        ]
        
        for feature, description in significant:
            if data.get(feature) in [1, 't', True]:
                findings.append(description)
        
        # Top contributing features from model
        if prediction.feature_contributions:
            top_feature = list(prediction.feature_contributions.keys())[0]
            findings.append(f"Primary risk driver: {top_feature}")
        
        return findings if findings else ["No significant abnormal findings"]
    
    def _create_recommendations(self, risk_level: str, evidence: List[RetrievedEvidence]) -> List[str]:
        """Create clinical recommendations"""
        base_recommendations = {
            'Low': [
                "Continue routine primary care monitoring",
                "Repeat thyroid function tests in 6-12 months",
                "Patient education on thyroid symptoms",
                "No immediate specialist referral required"
            ],
            'Moderate': [
                "Complete comprehensive thyroid panel (TSH, free T4, T3)",
                "Repeat testing in 4-6 weeks",
                "Consider endocrinology referral if tests remain abnormal",
                "Monitor symptoms closely"
            ],
            'High': [
                "URGENT: Complete comprehensive thyroid workup",
                "Endocrinology referral within 1-2 weeks",
                "Consider thyroid ultrasound",
                "Close follow-up with frequent testing",
                "Initiate treatment per specialist guidance"
            ]
        }
        
        recommendations = base_recommendations.get(risk_level, base_recommendations['Moderate'])
        
        # Add evidence-based recommendations
        if evidence:
            recommendations.append(
                f"See evidence from {evidence[0].citation_id} for detailed protocols"
            )
        
        return recommendations
    
    def _format_evidence_citations(self, evidence: List[RetrievedEvidence]) -> str:
        """Format evidence citations for doctor summary"""
        if not evidence:
            return "No specific guideline citations retrieved."
        
        citations = []
        for i, ev in enumerate(evidence, 1):
            citations.append(
                f"[{i}] {ev.citation_id} - {ev.title} (Relevance: {ev.relevance_score:.3f})"
            )
        
        return "\n".join(citations)
    
    def _create_doctor_next_steps(
        self, 
        prediction: RiskPrediction,
        data: Dict,
        reasoning: ClinicalReasoning
    ) -> str:
        """Create actionable next steps for doctor"""
        steps = []
        
        # Immediate actions
        steps.append("Immediate Actions:")
        if prediction.risk_level.value == "High":
            steps.append("  1. Order comprehensive thyroid panel immediately")
            steps.append("  2. Initiate urgent endocrinology referral")
            steps.append("  3. Document all symptoms and findings")
        elif prediction.risk_level.value == "Moderate":
            steps.append("  1. Complete thyroid function testing if not done")
            steps.append("  2. Schedule follow-up in 4-6 weeks")
            steps.append("  3. Provide patient education materials")
        else:
            steps.append("  1. Reassure patient")
            steps.append("  2. Schedule routine follow-up in 6-12 months")
            steps.append("  3. Educate on symptoms requiring earlier evaluation")
        
        # Missing tests
        missing_tests = [flag for flag in prediction.uncertainty_flags 
                        if 'not measured' in flag.lower()]
        if missing_tests:
            steps.append("\nRecommended Additional Testing:")
            for test in missing_tests:
                steps.append(f"  - {test}")
        
        return "\n".join(steps)
    
    # Patient Summary Components
    
    def _create_patient_overview(self, prediction: RiskPrediction) -> str:
        """Create patient-friendly overview"""
        risk_level = prediction.risk_level.value.lower()
        
        if risk_level == "low":
            overview = (
                "Your thyroid test results and health information have been reviewed. "
                "The analysis suggests you are at low risk for significant thyroid problems. "
                "This is good news and means routine follow-up should be sufficient."
            )
        elif risk_level == "moderate":
            overview = (
                "Your thyroid test results and health information have been reviewed. "
                "The analysis suggests you may need some additional thyroid evaluation. "
                "This doesn't necessarily mean you have a serious problem, but your doctor "
                "may want to run some additional tests to get a clearer picture."
            )
        else:  # high
            overview = (
                "Your thyroid test results and health information have been reviewed. "
                "The analysis suggests you should have a more detailed thyroid evaluation soon. "
                "Your doctor may refer you to a thyroid specialist to ensure you receive "
                "the best possible care."
            )
        
        return overview
    
    def _explain_to_patient(self, prediction: RiskPrediction, data: Dict) -> str:
        """Explain what the results mean in plain language"""
        explanations = []
        
        # Explain based on findings
        has_abnormal_labs = any(
            data.get(test) in [1, 't', True] 
            for test in ['TSH', 'T3', 'TT4', 'FTI']
        )
        
        if has_abnormal_labs:
            explanations.append(
                "Some of your thyroid test results are outside the normal range. "
                "This could indicate that your thyroid is either working too hard or not hard enough."
            )
        
        # Medication context
        on_medication = (
            data.get('on_thyroxine') in [1, 't', True] or 
            data.get('on_antithyroid_medication') in [1, 't', True]
        )
        
        if on_medication:
            explanations.append(
                "Since you're currently taking thyroid medication, these tests help your doctor "
                "make sure your treatment is working well and adjust it if needed."
            )
        
        # Surgery history
        if data.get('thyroid_surgery') in [1, 't', True]:
            explanations.append(
                "Because you've had thyroid surgery in the past, regular monitoring is "
                "an important part of your ongoing care."
            )
        
        if not explanations:
            explanations.append(
                "The evaluation looks at various factors including your test results, "
                "medical history, and current health status to help guide your care."
            )
        
        return " ".join(explanations)
    
    def _create_patient_next_steps(self, prediction: RiskPrediction) -> str:
        """Explain next steps in patient-friendly language"""
        risk_level = prediction.risk_level.value.lower()
        
        if risk_level == "low":
            steps = (
                "Your doctor will likely recommend:\n"
                "• Continue with your regular checkups\n"
                "• Repeat thyroid tests in 6-12 months\n"
                "• Watch for any new symptoms and report them to your doctor\n"
                "• Maintain a healthy lifestyle"
            )
        elif risk_level == "moderate":
            steps = (
                "Your doctor will likely recommend:\n"
                "• Some additional thyroid tests in the next few weeks\n"
                "• A follow-up appointment to review the results\n"
                "• Possibly a referral to a thyroid specialist if needed\n"
                "• Keep track of any symptoms you experience"
            )
        else:  # high
            steps = (
                "Your doctor will likely recommend:\n"
                "• Complete thyroid testing as soon as possible\n"
                "• A referral to a thyroid specialist (endocrinologist)\n"
                "• Close monitoring of your symptoms\n"
                "• Possible imaging studies like an ultrasound\n"
                "• Starting or adjusting treatment as recommended by the specialist"
            )
        
        return steps
    
    def _generate_patient_questions(self, prediction: RiskPrediction, data: Dict) -> List[str]:
        """Generate helpful questions patients should ask their doctor"""
        questions = [
            "What do my specific test results mean?",
            "What symptoms should I watch for?",
            "When should I schedule my next appointment?"
        ]
        
        if prediction.risk_level.value != "Low":
            questions.extend([
                "Do I need to see a specialist?",
                "What additional tests might I need?",
                "Are there any lifestyle changes I should make?"
            ])
        
        if data.get('on_thyroxine') or data.get('on_antithyroid_medication'):
            questions.append("Should my medication dosage be adjusted?")
        
        return questions
    
    def _create_reassurance(self, prediction: RiskPrediction) -> str:
        """Provide appropriate reassurance to patient"""
        risk_level = prediction.risk_level.value.lower()
        
        if risk_level == "low":
            reassurance = (
                "Remember, this assessment is based on your current information and is meant to help "
                "guide your care. Having low risk is encouraging, but it's still important to attend "
                "your regular checkups and communicate with your doctor about any concerns."
            )
        elif risk_level == "moderate":
            reassurance = (
                "While you may need additional testing, this doesn't automatically mean you have "
                "a serious thyroid condition. Many thyroid issues are very treatable, and catching "
                "them early leads to better outcomes. Your doctor is taking a careful, thorough "
                "approach to ensure you get the right care."
            )
        else:  # high
            reassurance = (
                "While being in a higher risk category may sound concerning, it's important to "
                "know that thyroid conditions are generally very treatable when properly managed. "
                "Your healthcare team is being appropriately cautious and thorough. Following their "
                "recommendations and attending appointments will help ensure you receive the best care."
            )
        
        reassurance += (
            "\n\nThis tool is designed to assist your doctor, not replace their expertise. "
            "Your doctor will consider this information along with their clinical judgment "
            "and your individual circumstances to provide you with personalized care."
        )
        
        return reassurance
    
    # Formatting Functions
    
    def format_doctor_summary(self, summary: DoctorSummary) -> str:
        """Format doctor summary as structured report"""
        report = []
        
        report.append("=" * 80)
        report.append("CLINICAL DECISION SUPPORT SUMMARY - FOR HEALTHCARE PROVIDER")
        report.append("=" * 80)
        
        report.append("\nEXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(summary.executive_summary)
        
        report.append("\n\nRISK ASSESSMENT")
        report.append("-" * 80)
        report.append(summary.risk_assessment)
        
        report.append("\n\nKEY CLINICAL FINDINGS")
        report.append("-" * 80)
        for i, finding in enumerate(summary.key_findings, 1):
            report.append(f"{i}. {finding}")
        
        report.append("\n\nRECOMMENDATIONS")
        report.append("-" * 80)
        for i, rec in enumerate(summary.recommendations, 1):
            report.append(f"{i}. {rec}")
        
        report.append("\n\nEVIDENCE CITATIONS")
        report.append("-" * 80)
        report.append(summary.evidence_citations)
        
        report.append("\n\nNEXT STEPS")
        report.append("-" * 80)
        report.append(summary.next_steps)
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def format_patient_summary(self, summary: PatientSummary) -> str:
        """Format patient summary in friendly, accessible language"""
        report = []
        
        report.append("=" * 80)
        report.append("YOUR THYROID HEALTH ASSESSMENT SUMMARY")
        report.append("=" * 80)
        
        report.append("\nOVERVIEW")
        report.append("-" * 80)
        report.append(summary.overview)
        
        report.append("\n\nWHAT THIS MEANS FOR YOU")
        report.append("-" * 80)
        report.append(summary.what_this_means)
        
        report.append("\n\nNEXT STEPS")
        report.append("-" * 80)
        report.append(summary.next_steps)
        
        report.append("\n\nQUESTIONS TO ASK YOUR DOCTOR")
        report.append("-" * 80)
        for i, question in enumerate(summary.questions_to_ask, 1):
            report.append(f"{i}. {question}")
        
        report.append("\n\nIMPORTANT INFORMATION")
        report.append("-" * 80)
        report.append(summary.reassurance)
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    agent = SummaryAgent()
    print("Summary Agent initialized successfully")
