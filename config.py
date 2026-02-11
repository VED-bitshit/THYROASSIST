"""
Configuration file for Thyroid Risk Assessment Multi-Agent System
"""

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

# Risk Categories
class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

# System Configuration
@dataclass
class ModelConfig:
    """Configuration for ML models"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    risk_threshold_moderate: float = 0.4
    risk_threshold_high: float = 0.7

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrievals: int = 3

# Data Schema
DEMOGRAPHIC_FEATURES = ['age', 'sex']

CLINICAL_HISTORY_FEATURES = [
    'on_thyroxine',
    'on_antithyroid_medication',
    'sick',
    'pregnant',
    'thyroid_surgery',
    'lithium',
    'goitre',
    'tumor',
    'hypopituitary',
    'psych'
]

LAB_MEASUREMENT_FEATURES = [
    'TSH_measured', 'TSH',
    'T3_measured', 'T3',
    'TT4_measured', 'TT4',
    'T4U_measured', 'T4U',
    'FTI_measured', 'FTI'
]

REFERRAL_FEATURES = ['referral_source']

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES + 
    CLINICAL_HISTORY_FEATURES + 
    LAB_MEASUREMENT_FEATURES + 
    REFERRAL_FEATURES
)

# Medical Guidelines Database (Sample)
MEDICAL_GUIDELINES = [
    {
        "id": "WHO_2023_001",
        "title": "WHO Thyroid Disorder Management Guidelines 2023",
        "content": """Patients with abnormal TSH levels should undergo comprehensive thyroid function testing 
        including free T4 and T3 measurements. For patients with a history of thyroid surgery or those 
        on thyroid medication, regular monitoring every 6-12 months is recommended. Immediate endocrinology 
        referral is indicated for TSH levels outside the reference range with concurrent symptoms."""
    },
    {
        "id": "ATA_2022_003",
        "title": "American Thyroid Association Clinical Practice Guidelines",
        "content": """Risk stratification should consider patient demographics, clinical history, and laboratory 
        findings. High-risk patients include those with: previous thyroid surgery, family history of thyroid 
        disorders, abnormal thyroid function tests, or symptoms of thyroid dysfunction. Such patients require 
        specialized evaluation and may benefit from imaging studies."""
    },
    {
        "id": "ENDOCRINE_2023_005",
        "title": "Endocrine Society Guidelines for Thyroid Testing",
        "content": """TSH is the primary screening test for thyroid dysfunction. If TSH is abnormal, measure 
        free T4. In patients on antithyroid medication or thyroxine, both TSH and free T4 should be measured. 
        T3 measurement is indicated when T3 thyrotoxicosis is suspected. FTI (Free Thyroxine Index) can be 
        used when direct free T4 measurement is unavailable."""
    },
    {
        "id": "CLINICAL_2023_012",
        "title": "Clinical Protocols for Thyroid Risk Assessment",
        "content": """Patients presenting with goiter, tumor history, or hypopituitary conditions require 
        comprehensive thyroid evaluation. Pregnant patients with thyroid dysfunction need specialized 
        management due to altered thyroid hormone requirements. Lithium therapy can affect thyroid function 
        and requires regular monitoring."""
    },
    {
        "id": "TRIAGE_2023_008",
        "title": "Emergency Department Thyroid Triage Protocol",
        "content": """Low-risk patients with normal TSH and no concerning symptoms can be managed in primary 
        care with routine follow-up. Moderate-risk patients should have repeat testing in 4-6 weeks. 
        High-risk patients with multiple abnormal findings, severe symptoms, or critical test results 
        require urgent endocrinology consultation within 1-2 weeks."""
    }
]

# Ethical Disclaimer
ETHICAL_DISCLAIMER = """
IMPORTANT MEDICAL DISCLAIMER:
- This is a clinical decision support tool, NOT a diagnostic system
- Results should be interpreted by qualified healthcare professionals
- This system provides recommendations based on available data and may be incomplete
- Clinicians must use their professional judgment and follow institutional protocols
- All patient data must be handled according to HIPAA/GDPR privacy standards
- This tool does not replace clinical examination or professional medical advice
"""
