"""
Multi-Agent Clinical Decision Support System Orchestrator
Coordinates all four agents to provide complete thyroid risk assessment
"""

import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
import json

from config import ETHICAL_DISCLAIMER
from agent1_risk_scoring import RiskScoringAgent, RiskPrediction
from agent2_knowledge_retriever import MedicalKnowledgeRetriever, RetrievedEvidence
from agent3_reasoning import ReasoningAgent, ClinicalReasoning
from agent4_summary import SummaryAgent, DoctorSummary, PatientSummary


@dataclass
class ClinicalAssessment:
    """Complete clinical assessment output"""
    risk_prediction: RiskPrediction
    retrieved_evidence: list
    clinical_reasoning: ClinicalReasoning
    doctor_summary: DoctorSummary
    patient_summary: PatientSummary
    ethical_disclaimer: str


class ThyroidClinicalDecisionSupport:
    """
    Main orchestrator for the multi-agent thyroid risk assessment system
    Coordinates all four agents to provide comprehensive clinical decision support
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize all agents
        
        Args:
            model_path: Path to trained ML model (optional, will train if not provided)
        """
        print("Initializing Thyroid Clinical Decision Support System...")
        print("-" * 80)
        
        # Initialize all agents
        print("Loading Agent 1: Risk Scoring Agent...")
        self.risk_agent = RiskScoringAgent()
        if model_path:
            self.risk_agent.load_model(model_path)
            print(f"  ✓ Loaded trained model from {model_path}")
        else:
            print("  ⚠ No trained model loaded. Train model before assessment.")
        
        print("\nLoading Agent 2: Medical Knowledge Retriever...")
        self.knowledge_agent = MedicalKnowledgeRetriever()
        print("  ✓ RAG system initialized")
        
        print("\nLoading Agent 3: Reasoning Agent...")
        self.reasoning_agent = ReasoningAgent()
        print("  ✓ Explainability system ready")
        
        print("\nLoading Agent 4: Summary Agent...")
        self.summary_agent = SummaryAgent()
        print("  ✓ Summary generation ready")
        
        print("-" * 80)
        print("System initialization complete!\n")
    
    def train_system(self, training_data_path: str, save_model_path: str = None):
        """
        Train the ML models
        
        Args:
            training_data_path: Path to training CSV file
            save_model_path: Path to save trained model
        """
        print("\n" + "=" * 80)
        print("TRAINING PHASE")
        print("=" * 80)
        
        print(f"\nTraining models using data from: {training_data_path}")
        results = self.risk_agent.train_models(training_data_path)
        
        print("\n" + "-" * 80)
        print("Training Results Summary:")
        print(f"  Best Model: {results['best_model']}")
        print(f"  Cross-validation AUC: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        print("-" * 80)
        
        if save_model_path:
            self.risk_agent.save_model(save_model_path)
            print(f"\nModel saved to: {save_model_path}")
        
        return results
    
    def assess_patient(self, patient_data: Dict) -> ClinicalAssessment:
        """
        Perform complete clinical assessment for a patient
        
        Args:
            patient_data: Dictionary or DataFrame with patient information
            
        Returns:
            ClinicalAssessment object with all results
        """
        print("\n" + "=" * 80)
        print("PATIENT ASSESSMENT")
        print("=" * 80)
        
        # Convert dict to DataFrame if necessary
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
            patient_dict = patient_data
        else:
            patient_df = patient_data
            patient_dict = patient_data.to_dict('records')[0]
        
        # STEP 1: Risk Scoring
        print("\n[1/4] Risk Scoring Agent - Calculating risk score...")
        risk_prediction = self.risk_agent.predict(patient_df)
        print(f"  ✓ Risk Level: {risk_prediction.risk_level.value}")
        print(f"  ✓ Risk Score: {risk_prediction.risk_score:.3f}")
        
        # STEP 2: Knowledge Retrieval
        print("\n[2/4] Medical Knowledge Retriever - Fetching relevant guidelines...")
        retrieved_evidence = self.knowledge_agent.retrieve_for_patient(
            risk_prediction, patient_dict
        )
        print(f"  ✓ Retrieved {len(retrieved_evidence)} relevant guidelines")
        
        # STEP 3: Clinical Reasoning
        print("\n[3/4] Reasoning Agent - Generating clinical explanation...")
        clinical_reasoning = self.reasoning_agent.generate_reasoning(
            risk_prediction, patient_dict, retrieved_evidence
        )
        print("  ✓ Clinical reasoning generated")
        
        # STEP 4: Summary Generation
        print("\n[4/4] Summary Agent - Creating summaries...")
        doctor_summary = self.summary_agent.generate_doctor_summary(
            risk_prediction, patient_dict, clinical_reasoning, retrieved_evidence
        )
        patient_summary = self.summary_agent.generate_patient_summary(
            risk_prediction, patient_dict, clinical_reasoning
        )
        print("  ✓ Doctor summary created")
        print("  ✓ Patient summary created")
        
        print("\n" + "=" * 80)
        print("Assessment Complete!")
        print("=" * 80)
        
        return ClinicalAssessment(
            risk_prediction=risk_prediction,
            retrieved_evidence=retrieved_evidence,
            clinical_reasoning=clinical_reasoning,
            doctor_summary=doctor_summary,
            patient_summary=patient_summary,
            ethical_disclaimer=ETHICAL_DISCLAIMER
        )
    
    def generate_report(self, assessment: ClinicalAssessment, 
                       include_doctor: bool = True,
                       include_patient: bool = True,
                       include_reasoning: bool = True) -> str:
        """
        Generate comprehensive report
        
        Args:
            assessment: ClinicalAssessment object
            include_doctor: Include doctor summary
            include_patient: Include patient summary
            include_reasoning: Include detailed reasoning
            
        Returns:
            Formatted report string
        """
        report_sections = []
        
        # Disclaimer
        report_sections.append(assessment.ethical_disclaimer)
        report_sections.append("\n" + "=" * 80 + "\n")
        
        # Doctor Summary
        if include_doctor:
            report_sections.append(
                self.summary_agent.format_doctor_summary(assessment.doctor_summary)
            )
            report_sections.append("\n")
        
        # Detailed Reasoning
        if include_reasoning:
            report_sections.append(
                self.reasoning_agent.format_reasoning_report(assessment.clinical_reasoning)
            )
            report_sections.append("\n")
        
        # Patient Summary
        if include_patient:
            report_sections.append(
                self.summary_agent.format_patient_summary(assessment.patient_summary)
            )
            report_sections.append("\n")
        
        # Citations
        report_sections.append("=" * 80)
        report_sections.append("MEDICAL GUIDELINE CITATIONS")
        report_sections.append("=" * 80)
        report_sections.append(
            self.knowledge_agent.format_citations(assessment.retrieved_evidence)
        )
        
        return "\n".join(report_sections)
    
    def save_assessment(self, assessment: ClinicalAssessment, filepath: str):
        """Save assessment to JSON file"""
        assessment_dict = {
            'risk_prediction': {
                'risk_score': assessment.risk_prediction.risk_score,
                'risk_level': assessment.risk_prediction.risk_level.value,
                'confidence_interval': [
                    assessment.risk_prediction.confidence_lower,
                    assessment.risk_prediction.confidence_upper
                ],
                'uncertainty_flags': assessment.risk_prediction.uncertainty_flags,
                'model_name': assessment.risk_prediction.model_name
            },
            'evidence_count': len(assessment.retrieved_evidence),
            'evidence_citations': [
                {
                    'citation_id': ev.citation_id,
                    'title': ev.title,
                    'relevance_score': ev.relevance_score
                }
                for ev in assessment.retrieved_evidence
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(assessment_dict, f, indent=2)
        
        print(f"\nAssessment saved to: {filepath}")


# Command-line interface
def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Thyroid Clinical Decision Support System'
    )
    parser.add_argument(
        '--train',
        type=str,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='thyroid_model.pkl',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--assess',
        type=str,
        help='Path to patient data CSV file for assessment'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='assessment_report.txt',
        help='Path for output report'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    if args.train:
        system = ThyroidClinicalDecisionSupport()
        system.train_system(args.train, args.model)
    
    if args.assess:
        system = ThyroidClinicalDecisionSupport(model_path=args.model)
        
        # Load patient data
        patient_data = pd.read_csv(args.assess)
        
        # Assess first patient (or loop through all)
        assessment = system.assess_patient(patient_data.iloc[0])
        
        # Generate and save report
        report = system.generate_report(assessment)
        
        with open(args.output, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {args.output}")
        
        # Also save JSON
        json_output = args.output.replace('.txt', '.json')
        system.save_assessment(assessment, json_output)


if __name__ == "__main__":
    # Example usage without command line
  
    system = ThyroidClinicalDecisionSupport()
    system.train_system('/Users/saadhanagroup/Downloads/AGENTX/Thyroid_Data.csv', 'model.pkl')
    
   
