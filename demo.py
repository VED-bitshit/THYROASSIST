"""
Demo Script: End-to-End Example of Thyroid Clinical Decision Support System
Demonstrates how to use all four agents together
"""

import pandas as pd
import numpy as np
from main_system import ThyroidClinicalDecisionSupport


def create_sample_patient_data():
    """Create sample patient data for demonstration"""
    
    # Sample Patient 1: High Risk
    patient_high_risk = {
        'age': 65,
        'sex': 'f',
        'on_thyroxine': 'f',
        'on_antithyroid_medication': 't',
        'sick': 'f',
        'pregnant': 'f',
        'thyroid_surgery': 't',
        'lithium': 'f',
        'goitre': 't',
        'tumor': 'f',
        'hypopituitary': 'f',
        'psych': 'f',
        'TSH_measured': 't',
        'TSH': 't',  # abnormal
        'T3_measured': 't',
        'T3': 't',  # abnormal
        'TT4_measured': 't',
        'TT4': 'f',
        'T4U_measured': 't',
        'T4U': 'f',
        'FTI_measured': 't',
        'FTI': 'f',
        'referral_source': 'SVHC'
    }
    
    # Sample Patient 2: Low Risk
    patient_low_risk = {
        'age': 35,
        'sex': 'm',
        'on_thyroxine': 'f',
        'on_antithyroid_medication': 'f',
        'sick': 'f',
        'pregnant': 'f',
        'thyroid_surgery': 'f',
        'lithium': 'f',
        'goitre': 'f',
        'tumor': 'f',
        'hypopituitary': 'f',
        'psych': 'f',
        'TSH_measured': 't',
        'TSH': 'f',  # normal
        'T3_measured': 't',
        'T3': 'f',  # normal
        'TT4_measured': 't',
        'TT4': 'f',
        'T4U_measured': 't',
        'T4U': 'f',
        'FTI_measured': 't',
        'FTI': 'f',
        'referral_source': 'other'
    }
    
    # Sample Patient 3: Moderate Risk with Missing Data
    patient_moderate_risk = {
        'age': 52,
        'sex': 'f',
        'on_thyroxine': 't',
        'on_antithyroid_medication': 'f',
        'sick': 'f',
        'pregnant': 'f',
        'thyroid_surgery': 'f',
        'lithium': 'f',
        'goitre': 'f',
        'tumor': 'f',
        'hypopituitary': 'f',
        'psych': 'f',
        'TSH_measured': 't',
        'TSH': 't',  # abnormal
        'T3_measured': 'f',  # not measured
        'T3': 'f',
        'TT4_measured': 't',
        'TT4': 'f',
        'T4U_measured': 'f',  # not measured
        'T4U': 'f',
        'FTI_measured': 't',
        'FTI': 'f',
        'referral_source': 'SVI'
    }
    
    return [patient_high_risk, patient_low_risk, patient_moderate_risk]


def demo_without_training():
    """
    Demonstrate system functionality without actual training
    (Uses mock predictions for demonstration purposes)
    """
    print("=" * 80)
    print("THYROID CLINICAL DECISION SUPPORT SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print("\nNOTE: This demo uses simulated predictions since no training data is available.")
    print("In production, you would first train the system on actual thyroid data.\n")
    
    # Get sample patients
    patients = create_sample_patient_data()
    patient_names = ["High-Risk Patient", "Low-Risk Patient", "Moderate-Risk Patient"]
    
    # Initialize system (without trained model for demo)
    from agent2_knowledge_retriever import MedicalKnowledgeRetriever
    from agent3_reasoning import ReasoningAgent
    from agent4_summary import SummaryAgent
    
    knowledge_agent = MedicalKnowledgeRetriever()
    reasoning_agent = ReasoningAgent()
    summary_agent = SummaryAgent()
    
    # Process each patient
    for patient_data, patient_name in zip(patients, patient_names):
        print("\n" + "=" * 80)
        print(f"ASSESSING: {patient_name}")
        print("=" * 80)
        
        # Display patient info
        print("\nPatient Information:")
        print(f"  Age: {patient_data['age']}")
        print(f"  Sex: {patient_data['sex'].upper()}")
        
        # Clinical history
        clinical_flags = []
        if patient_data['on_thyroxine'] == 't':
            clinical_flags.append("On thyroxine")
        if patient_data['on_antithyroid_medication'] == 't':
            clinical_flags.append("On antithyroid medication")
        if patient_data['thyroid_surgery'] == 't':
            clinical_flags.append("History of thyroid surgery")
        if patient_data['goitre'] == 't':
            clinical_flags.append("Goiter present")
        
        if clinical_flags:
            print(f"  Clinical History: {', '.join(clinical_flags)}")
        
        # Lab results
        lab_results = []
        if patient_data['TSH'] == 't':
            lab_results.append("TSH abnormal")
        if patient_data['T3_measured'] == 't' and patient_data['T3'] == 't':
            lab_results.append("T3 abnormal")
        
        if lab_results:
            print(f"  Lab Results: {', '.join(lab_results)}")
        
        # Simulate knowledge retrieval
        print("\n[Agent 2] Retrieving relevant medical guidelines...")
        
        # Build query based on patient
        query_terms = []
        if patient_data['TSH'] == 't':
            query_terms.append("abnormal TSH")
        if patient_data['thyroid_surgery'] == 't':
            query_terms.append("thyroid surgery")
        if patient_data['on_antithyroid_medication'] == 't':
            query_terms.append("antithyroid medication")
        
        query = " ".join(query_terms) if query_terms else "thyroid assessment"
        evidence = knowledge_agent.retrieve(query, top_k=2)
        
        print(f"  Retrieved {len(evidence)} relevant guidelines:")
        for ev in evidence:
            print(f"    - {ev.citation_id}: {ev.title}")
        
        print("\n" + "-" * 80)
        print("MEDICAL EVIDENCE:")
        print("-" * 80)
        print(knowledge_agent.format_citations(evidence))
        
        print("\n" + "=" * 80)
        print(f"END OF ASSESSMENT: {patient_name}")
        print("=" * 80)


def demo_with_synthetic_training_data():
    """
    Create synthetic training data and demonstrate full system
    """
    print("=" * 80)
    print("FULL SYSTEM DEMONSTRATION WITH SYNTHETIC DATA")
    print("=" * 80)
    
    # Create synthetic training data
    print("\nCreating synthetic training dataset...")
    n_samples = 500
    
    # Generate random features
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'sex': np.random.choice(['m', 'f'], n_samples),
        'on_thyroxine': np.random.choice(['t', 'f'], n_samples, p=[0.3, 0.7]),
        'on_antithyroid_medication': np.random.choice(['t', 'f'], n_samples, p=[0.2, 0.8]),
        'sick': np.random.choice(['t', 'f'], n_samples, p=[0.1, 0.9]),
        'pregnant': np.random.choice(['t', 'f'], n_samples, p=[0.05, 0.95]),
        'thyroid_surgery': np.random.choice(['t', 'f'], n_samples, p=[0.15, 0.85]),
        'lithium': np.random.choice(['t', 'f'], n_samples, p=[0.05, 0.95]),
        'goitre': np.random.choice(['t', 'f'], n_samples, p=[0.2, 0.8]),
        'tumor': np.random.choice(['t', 'f'], n_samples, p=[0.1, 0.9]),
        'hypopituitary': np.random.choice(['t', 'f'], n_samples, p=[0.05, 0.95]),
        'psych': np.random.choice(['t', 'f'], n_samples, p=[0.1, 0.9]),
        'TSH_measured': np.random.choice(['t', 'f'], n_samples, p=[0.9, 0.1]),
        'TSH': np.random.choice(['t', 'f'], n_samples, p=[0.3, 0.7]),
        'T3_measured': np.random.choice(['t', 'f'], n_samples, p=[0.8, 0.2]),
        'T3': np.random.choice(['t', 'f'], n_samples, p=[0.2, 0.8]),
        'TT4_measured': np.random.choice(['t', 'f'], n_samples, p=[0.85, 0.15]),
        'TT4': np.random.choice(['t', 'f'], n_samples, p=[0.25, 0.75]),
        'T4U_measured': np.random.choice(['t', 'f'], n_samples, p=[0.7, 0.3]),
        'T4U': np.random.choice(['t', 'f'], n_samples, p=[0.15, 0.85]),
        'FTI_measured': np.random.choice(['t', 'f'], n_samples, p=[0.75, 0.25]),
        'FTI': np.random.choice(['t', 'f'], n_samples, p=[0.2, 0.8]),
        'referral_source': np.random.choice(['SVI', 'SVHC', 'other'], n_samples)
    }
    
    # Create synthetic target based on risk factors
    # Higher risk if: abnormal TSH, surgery history, on medication, abnormal labs
    risk_score = np.zeros(n_samples)
    
    for i in range(n_samples):
        score = 0
        if data['TSH'][i] == 't':
            score += 0.3
        if data['thyroid_surgery'][i] == 't':
            score += 0.2
        if data['on_antithyroid_medication'][i] == 't':
            score += 0.2
        if data['T3'][i] == 't':
            score += 0.15
        if data['goitre'][i] == 't':
            score += 0.15
        
        risk_score[i] = min(score + np.random.normal(0, 0.1), 1.0)
    
    # Create binary target (negative vs positive)
    data['class'] = ['positive' if score > 0.5 else 'negative' for score in risk_score]
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('/home/claude/synthetic_thyroid_data.csv', index=False)
    print(f"  ✓ Created {n_samples} synthetic patient records")
    print(f"  ✓ Saved to: synthetic_thyroid_data.csv")
    
    # Initialize and train system
    print("\nInitializing and training system...")
    system = ThyroidClinicalDecisionSupport()
    
    print("\nTraining models...")
    training_results = system.train_system(
        '/home/claude/synthetic_thyroid_data.csv',
        '/home/claude/thyroid_model.pkl'
    )
    
    # Assess sample patients
    print("\n" + "=" * 80)
    print("PATIENT ASSESSMENTS")
    print("=" * 80)
    
    patients = create_sample_patient_data()
    
    for i, patient_data in enumerate(patients[:2], 1):  # Assess first 2 patients
        print(f"\n{'=' * 80}")
        print(f"PATIENT {i} ASSESSMENT")
        print(f"{'=' * 80}")
        
        # Perform assessment
        assessment = system.assess_patient(patient_data)
        
        # Generate and display report
        report = system.generate_report(
            assessment,
            include_doctor=True,
            include_patient=True,
            include_reasoning=True
        )
        
        # Save report
        report_path = f'/home/claude/patient_{i}_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Full report saved to: {report_path}")
        
        # Display summary
        print("\n" + "-" * 80)
        print("QUICK SUMMARY:")
        print("-" * 80)
        print(f"Risk Level: {assessment.risk_prediction.risk_level.value}")
        print(f"Risk Score: {assessment.risk_prediction.risk_score:.3f}")
        print(f"Guidelines Retrieved: {len(assessment.retrieved_evidence)}")
        print("-" * 80)


if __name__ == "__main__":
    import sys
    
    print("""
    THYROID CLINICAL DECISION SUPPORT SYSTEM - DEMO
    ================================================
    
    Choose demo mode:
    1. Demo without training (knowledge retrieval only)
    2. Full system demo with synthetic data (includes ML training)
    
    """)
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        demo_without_training()
    elif choice == '2':
        demo_with_synthetic_training_data()
    else:
        print("Invalid choice. Running demo mode 1...")
        demo_without_training()
