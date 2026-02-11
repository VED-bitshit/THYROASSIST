# Quick Start Guide - Thyroid Clinical Decision Support System

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import pandas, sklearn, numpy; print('✓ Core dependencies installed')"
```

### Optional: Install sentence-transformers for better RAG
```bash
pip install sentence-transformers
```

## Running the Demo (2 minutes)

### Option 1: Quick Demo (No Training)
```bash
python demo.py
# Select option 1
```

This will:
- Demonstrate the knowledge retrieval system
- Show how medical guidelines are retrieved
- Display sample assessments

### Option 2: Full Demo (With Training)
```bash
python demo.py
# Select option 2
```

This will:
- Create synthetic training data (500 patients)
- Train ML models (Logistic Regression, Random Forest, Gradient Boosting)
- Select best model based on cross-validation
- Assess sample patients
- Generate complete reports

## Using With Your Own Data

### 1. Training Phase

```python
from main_system import ThyroidClinicalDecisionSupport

# Initialize system
system = ThyroidClinicalDecisionSupport()

# Train on your data
results = system.train_system(
    training_data_path='your_thyroid_data.csv',
    save_model_path='your_model.pkl'
)

print(f"Best model: {results['best_model']}")
print(f"AUC-ROC: {results['cv_mean']:.4f}")
```

### 2. Assessment Phase

```python
from main_system import ThyroidClinicalDecisionSupport

# Load trained model
system = ThyroidClinicalDecisionSupport(model_path='your_model.pkl')

# Assess a patient
patient = {
    'age': 55,
    'sex': 'f',
    'on_thyroxine': 't',
    'TSH_measured': 't',
    'TSH': 't',
    # ... other features
}

assessment = system.assess_patient(patient)

# Generate report
report = system.generate_report(assessment)
print(report)

# Save report
with open('patient_report.txt', 'w') as f:
    f.write(report)
```

## Command Line Usage

### Train the system:
```bash
python main_system.py --train Thyroid_Data.csv --model thyroid_model.pkl
```

### Assess a patient:
```bash
python main_system.py --assess patient_data.csv --model thyroid_model.pkl --output report.txt
```

## Using the Jupyter Notebook

```bash
jupyter notebook thyroid_system_demo.ipynb
```

The notebook includes:
- Interactive demonstrations
- Visualization of risk scores
- Step-by-step walkthrough of all agents
- Sample patient assessments

## System Architecture

```
Patient Data → Agent 1 (Risk Scoring) → Risk Score + Confidence
                    ↓
              Agent 2 (Knowledge Retrieval) → Medical Guidelines
                    ↓
              Agent 3 (Reasoning) → Transparent Explanation
                    ↓
              Agent 4 (Summary) → Doctor + Patient Summaries
                    ↓
              Complete Clinical Assessment
```

## Expected Data Format

Your CSV file should have these columns:

**Demographics:**
- age, sex

**Clinical History (t/f):**
- on_thyroxine, on_antithyroid_medication, sick, pregnant
- thyroid_surgery, lithium, goitre, tumor, hypopituitary, psych

**Lab Tests:**
- TSH_measured, TSH (and similarly for T3, TT4, T4U, FTI)

**Referral:**
- referral_source

**Target (for training only):**
- class (e.g., 'negative', 'positive')

## Output Files

The system generates:
- **Risk assessment** (score, level, confidence)
- **Doctor summary** (clinical, technical)
- **Patient summary** (plain language)
- **Clinical reasoning** (transparent explanation)
- **Evidence citations** (medical guidelines)

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution:** Install missing package
```bash
pip install <package_name>
```

### Issue: sentence-transformers not available
**Solution:** System will automatically use TF-IDF fallback (works fine for demo)

### Issue: No trained model
**Solution:** Train the system first using demo.py option 2 or main_system.py --train

## Next Steps

1. ✅ Run the demo to see the system in action
2. ✅ Explore the Jupyter notebook for interactive learning
3. ✅ Train with your own thyroid data
4. ✅ Customize medical guidelines in config.py
5. ✅ Adjust risk thresholds as needed
6. ✅ Add additional ML models if desired

## Important Notes

⚠️ **This is a decision support tool, NOT a diagnostic system**
- Results must be interpreted by qualified healthcare professionals
- Requires proper validation before clinical use
- Patient privacy must follow HIPAA/GDPR standards

## Support

For questions:
- Check README.md for detailed documentation
- Review source code comments
- Run demo.py for examples

---

**Remember:** Always validate with real clinical data before any medical application!
