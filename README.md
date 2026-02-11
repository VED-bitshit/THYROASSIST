# Thyroid Risk Assessment - Multi-Agent Clinical Decision Support System

## Overview

This is a comprehensive **Multi-Agent AI System** for thyroid-related risk assessment and clinical decision support. The system integrates **four specialized AI agents** that work together to provide evidence-based, explainable, and actionable clinical recommendations.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATIENT DATA INPUT                           │
│          (Demographics, Labs, Clinical History)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 1: Risk Scoring Agent                                    │
│  • ML-based risk prediction (Logistic Regression, Random        │
│    Forest, Gradient Boosting)                                   │
│  • Risk score + confidence interval                             │
│  • Uncertainty detection                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 2: Medical Knowledge Retriever (RAG)                     │
│  • Retrieval-Augmented Generation                               │
│  • Searches clinical guidelines (WHO, ATA, etc.)                │
│  • Context-aware evidence retrieval                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 3: Reasoning & Explainability Agent                      │
│  • Transparent reasoning chains                                 │
│  • Feature → Prediction → Evidence linkage                      │
│  • Uncertainty quantification                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 4: Summary Agent                                         │
│  • Doctor summary (technical, clinical)                         │
│  • Patient summary (plain language)                             │
│  • Actionable recommendations                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLINICAL DECISION SUPPORT                     │
│              (Risk Level, Evidence, Recommendations)            │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### ✅ Complete Multi-Agent Pipeline
- **Agent 1**: Machine learning risk scoring with confidence intervals
- **Agent 2**: RAG-based medical knowledge retrieval
- **Agent 3**: Transparent clinical reasoning and explainability
- **Agent 4**: Dual summaries (doctor + patient)

### ✅ Production-Ready ML Models
- Multiple algorithm comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Cross-validation and proper evaluation metrics (AUC-ROC, Precision, Recall, F1)
- Calibration analysis for reliable probability estimates
- Missing data handling and uncertainty detection

### ✅ Evidence-Based Recommendations
- Retrieves relevant clinical guidelines
- Provides citations for all recommendations
- Context-aware retrieval based on patient features

### ✅ Explainable AI
- Feature importance analysis
- Transparent reasoning chains
- Uncertainty quantification
- Clear limitation statements

### ✅ Safety & Ethics
- Clear disclaimer that this is decision support, not diagnosis
- Appropriate risk stratification
- Human-in-the-loop design philosophy

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install sentence-transformers for better RAG**:
```bash
pip install sentence-transformers
```
(Without this, the system will fall back to TF-IDF based retrieval)

## File Structure

```
thyroid-clinical-support/
│
├── config.py                    # System configuration and constants
├── preprocessing.py             # Data preprocessing utilities
│
├── agent1_risk_scoring.py       # Agent 1: ML Risk Prediction
├── agent2_knowledge_retriever.py # Agent 2: RAG System
├── agent3_reasoning.py          # Agent 3: Explainability
├── agent4_summary.py            # Agent 4: Summary Generation
│
├── main_system.py               # Main orchestrator
├── demo.py                      # Demonstration script
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

### Quick Start Demo

Run the demonstration script to see the system in action:

```bash
python demo.py
```

Choose between:
1. **Demo without training**: Shows knowledge retrieval and reasoning without ML
2. **Full system demo**: Creates synthetic data, trains models, and runs complete assessments

### Training the System

To train on your own thyroid data:

```python
from main_system import ThyroidClinicalDecisionSupport

# Initialize system
system = ThyroidClinicalDecisionSupport()

# Train models
results = system.train_system(
    training_data_path='Thyroid_Data.csv',
    save_model_path='thyroid_model.pkl'
)

print(f"Best model: {results['best_model']}")
print(f"AUC-ROC: {results['cv_mean']:.4f}")
```

### Assessing Patients

Once trained, assess individual patients:

```python
from main_system import ThyroidClinicalDecisionSupport
import pandas as pd

# Load trained system
system = ThyroidClinicalDecisionSupport(model_path='thyroid_model.pkl')

# Patient data
patient_data = {
    'age': 55,
    'sex': 'f',
    'on_thyroxine': 't',
    'thyroid_surgery': 't',
    'TSH_measured': 't',
    'TSH': 't',  # abnormal
    # ... (other features)
}

# Perform assessment
assessment = system.assess_patient(patient_data)

# Generate report
report = system.generate_report(assessment)
print(report)

# Save report
with open('patient_report.txt', 'w') as f:
    f.write(report)
```

### Command Line Interface

Train the system:
```bash
python main_system.py --train Thyroid_Data.csv --model thyroid_model.pkl
```

Assess a patient:
```bash
python main_system.py --assess patient_data.csv --model thyroid_model.pkl --output report.txt
```

## Data Schema

The system expects patient data with the following structure:

### Demographics
- `age`: Numeric (years)
- `sex`: Categorical (m/f)

### Clinical History (binary: t/f)
- `on_thyroxine`: Currently on thyroxine medication
- `on_antithyroid_medication`: Currently on antithyroid medication
- `sick`: Currently sick
- `pregnant`: Currently pregnant
- `thyroid_surgery`: History of thyroid surgery
- `lithium`: On lithium therapy
- `goitre`: Goiter present
- `tumor`: Thyroid tumor present
- `hypopituitary`: Hypopituitary condition
- `psych`: Psychiatric condition

### Laboratory Tests (measured + value)
For each test, two fields:
- `{TEST}_measured`: Whether test was performed (t/f)
- `{TEST}`: Whether value is abnormal (t/f)

Tests: TSH, T3, TT4, T4U, FTI

### Referral Context
- `referral_source`: Source of referral (SVI/SVHC/other)

## System Outputs

For each patient, the system generates:

### 1. Risk Assessment
- Risk score (0-1 probability)
- Risk level (Low/Moderate/High)
- Confidence interval
- Uncertainty flags

### 2. Doctor Summary
- Executive summary
- Risk assessment details
- Key clinical findings
- Evidence-based recommendations
- Clinical pathway
- Guideline citations

### 3. Patient Summary
- Plain language overview
- Explanation of findings
- Next steps
- Questions to ask doctor
- Appropriate reassurance

### 4. Clinical Reasoning
- Risk score justification
- Feature analysis
- Evidence synthesis
- Uncertainty statement
- Limitations
- Recommended pathway

## Evaluation Metrics

The ML models are evaluated using:

- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Calibration**: How well predicted probabilities match actual outcomes
- **Cross-Validation**: K-fold validation to assess generalization

## Medical Guidelines Database

The system includes curated medical guidelines from:
- WHO Thyroid Management Guidelines
- American Thyroid Association (ATA) Guidelines
- Endocrine Society Guidelines
- Clinical Triage Protocols

The RAG system retrieves the most relevant guidelines based on patient context.

## Ethical Considerations

### Important Disclaimers

⚠️ **This is a clinical decision support tool, NOT a diagnostic system**

- Results must be interpreted by qualified healthcare professionals
- Clinicians must use professional judgment and institutional protocols
- System outputs depend on input data quality
- Patient privacy must follow HIPAA/GDPR standards

### Safety Features

- Explicit uncertainty quantification
- Clear statement of limitations
- Evidence-based recommendations only
- Appropriate risk stratification
- No overconfident predictions

## Customization

### Adding New Medical Guidelines

Edit `config.py` and add to `MEDICAL_GUIDELINES`:

```python
{
    "id": "YOUR_GUIDELINE_ID",
    "title": "Guideline Title",
    "content": "Full guideline text..."
}
```

### Adjusting Risk Thresholds

Modify in `config.py`:

```python
class ModelConfig:
    risk_threshold_moderate: float = 0.4  # Adjust threshold
    risk_threshold_high: float = 0.7      # Adjust threshold
```

### Using Different ML Models

In `agent1_risk_scoring.py`, add your preferred model:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Add to training pipeline
your_model = YourModel(...)
self.models['your_model'] = your_model
```

## Testing

Run the demo with synthetic data to verify installation:

```bash
python demo.py
# Select option 2
```

## Contributing

This system is designed for educational and research purposes. When adapting for clinical use:

1. Validate with real clinical data
2. Obtain appropriate regulatory approvals
3. Implement proper data security
4. Conduct clinical validation studies
5. Ensure proper medical oversight

## License

This code is provided for educational and research purposes. Medical applications require appropriate validation and regulatory approval.

## Citation

If you use this system in research, please cite appropriately and acknowledge that this is a decision support tool requiring clinical validation.

## Support

For questions about implementation or customization, refer to:
- Individual agent source code files
- Inline documentation and comments
- Demo script for usage examples

---

**Remember**: This system is designed to ASSIST healthcare providers, not replace them. All clinical decisions should be made by qualified medical professionals using their expertise and judgment.
