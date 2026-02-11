# System Architecture Documentation

## Overview

The Thyroid Clinical Decision Support System implements a **multi-agent architecture** where four specialized AI agents collaborate to provide comprehensive clinical assessments.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│                                                                 │
│  Patient Data: Demographics, Clinical History, Lab Results     │
│  Format: CSV, Dict, DataFrame                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Preprocessing & Validation
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING MODULE                         │
│                                                                 │
│  • Handle missing values (imputation)                           │
│  • Encode categorical variables                                │
│  • Normalize numerical features                                │
│  • Validate data consistency                                   │
│  • Generate metadata (missing tests, quality scores)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               AGENT 1: RISK SCORING AGENT                       │
│                                                                 │
│  ML Models:                                                     │
│    • Logistic Regression                                       │
│    • Random Forest Classifier                                  │
│    • Gradient Boosting Classifier                              │
│                                                                 │
│  Training Phase:                                                │
│    • Train/test split (80/20)                                  │
│    • Cross-validation (5-fold)                                 │
│    • Model selection (best AUC-ROC)                            │
│    • Calibration analysis                                      │
│                                                                 │
│  Prediction Phase:                                              │
│    • Risk score (0-1 probability)                              │
│    • Risk level (Low/Moderate/High)                            │
│    • Confidence interval                                       │
│    • Feature importance                                        │
│    • Uncertainty flags                                         │
│                                                                 │
│  Output:                                                        │
│    RiskPrediction {                                             │
│      risk_score: float                                          │
│      risk_level: RiskLevel                                      │
│      confidence_lower: float                                    │
│      confidence_upper: float                                    │
│      uncertainty_flags: List[str]                               │
│      feature_contributions: Dict[str, float]                    │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          AGENT 2: MEDICAL KNOWLEDGE RETRIEVER (RAG)             │
│                                                                 │
│  Knowledge Base:                                                │
│    • WHO Thyroid Guidelines                                    │
│    • American Thyroid Association (ATA) Protocols              │
│    • Endocrine Society Guidelines                              │
│    • Clinical Triage Protocols                                 │
│                                                                 │
│  Retrieval Methods:                                             │
│    Option 1: Semantic Search (sentence-transformers)           │
│      • Encode guidelines as embeddings                         │
│      • Encode query as embedding                               │
│      • Cosine similarity matching                              │
│                                                                 │
│    Option 2: TF-IDF (fallback)                                 │
│      • Vector space model                                      │
│      • Term frequency analysis                                 │
│      • Cosine similarity                                       │
│                                                                 │
│  Query Enhancement:                                             │
│    • Enriched with patient context                             │
│    • Risk level integration                                    │
│    • Key clinical features                                     │
│    • Missing test information                                  │
│                                                                 │
│  Output:                                                        │
│    List[RetrievedEvidence] {                                    │
│      citation_id: str                                           │
│      title: str                                                 │
│      content: str                                               │
│      relevance_score: float                                     │
│      snippet: str                                               │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         AGENT 3: REASONING & EXPLAINABILITY AGENT               │
│                                                                 │
│  Reasoning Components:                                          │
│                                                                 │
│  1. Risk Justification                                          │
│     • Why this risk score was assigned                         │
│     • Top contributing features                                │
│     • Feature importance weights                               │
│                                                                 │
│  2. Feature Analysis                                            │
│     • Patient demographics summary                             │
│     • Clinical history review                                  │
│     • Laboratory findings analysis                             │
│                                                                 │
│  3. Evidence Synthesis                                          │
│     • Link predictions to guidelines                           │
│     • Cite supporting evidence                                 │
│     • Connect evidence to recommendations                      │
│                                                                 │
│  4. Uncertainty Quantification                                  │
│     • Confidence interval interpretation                       │
│     • Missing data impact                                      │
│     • Data quality assessment                                  │
│                                                                 │
│  5. Limitation Identification                                   │
│     • Missing critical tests                                   │
│     • Model limitations                                        │
│     • Scope boundaries                                         │
│                                                                 │
│  6. Clinical Pathway Recommendation                             │
│     • Immediate actions                                        │
│     • Follow-up testing schedule                               │
│     • Referral recommendations                                 │
│                                                                 │
│  Output:                                                        │
│    ClinicalReasoning {                                          │
│      risk_justification: str                                    │
│      feature_analysis: str                                      │
│      evidence_synthesis: str                                    │
│      uncertainty_statement: str                                 │
│      limitations: List[str]                                     │
│      clinical_pathway: str                                      │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AGENT 4: SUMMARY AGENT                          │
│                                                                 │
│  Generates Two Distinct Outputs:                                │
│                                                                 │
│  A) DOCTOR SUMMARY (Clinical/Technical)                         │
│     • Executive summary                                        │
│     • Risk assessment details                                  │
│     • Key clinical findings                                    │
│     • Evidence-based recommendations                           │
│     • Next steps with timeline                                 │
│     • Guideline citations                                      │
│                                                                 │
│     Output Format: Structured clinical report                  │
│     Tone: Professional, technical, actionable                  │
│                                                                 │
│  B) PATIENT SUMMARY (Plain Language)                            │
│     • Overview in simple terms                                 │
│     • What the results mean                                    │
│     • Next steps explained clearly                             │
│     • Questions to ask doctor                                  │
│     • Appropriate reassurance                                  │
│                                                                 │
│     Output Format: Friendly, accessible explanation            │
│     Tone: Supportive, non-alarming, empowering                 │
│                                                                 │
│  Output:                                                        │
│    {                                                            │
│      DoctorSummary: {...}                                       │
│      PatientSummary: {...}                                      │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│                                                                 │
│  Complete Clinical Assessment:                                  │
│    • Risk prediction                                           │
│    • Retrieved evidence                                        │
│    • Clinical reasoning                                        │
│    • Doctor summary                                            │
│    • Patient summary                                           │
│    • Ethical disclaimer                                        │
│                                                                 │
│  Output Formats:                                                │
│    • Structured text reports                                   │
│    • JSON for programmatic access                              │
│    • PDF (future enhancement)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Phase
```
Training Data (CSV)
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Model Training (3 algorithms)
    ↓
Cross-Validation
    ↓
Model Selection (best AUC-ROC)
    ↓
Save Model + Preprocessor
```

### 2. Assessment Phase
```
Patient Data
    ↓
Preprocessing (using saved preprocessor)
    ↓
Agent 1: Risk Prediction
    ↓
Agent 2: Evidence Retrieval (parallel)
    ↓
Agent 3: Reasoning Generation
    ↓
Agent 4: Summary Creation
    ↓
Complete Assessment Report
```

## Key Design Patterns

### 1. Modular Agent Architecture
Each agent is **independent and reusable**:
- Can be tested separately
- Can be enhanced independently
- Clear input/output contracts
- Easy to maintain and extend

### 2. Pipeline Orchestration
The `main_system.py` orchestrator:
- Coordinates all agents
- Manages data flow
- Handles error propagation
- Generates unified reports

### 3. Explainability-First Design
Every prediction includes:
- Why it was made (feature importance)
- Supporting evidence (citations)
- Confidence level (uncertainty)
- Limitations (what's missing)

### 4. Dual Audience Output
Separate summaries for:
- **Clinicians**: Technical, actionable, evidence-based
- **Patients**: Accessible, supportive, educational

## Technology Stack

### Core ML/Data Science
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **scikit-learn**: ML models and preprocessing

### NLP & RAG
- **sentence-transformers** (optional): Semantic embeddings
- **TF-IDF** (fallback): Text vectorization

### Model Persistence
- **joblib**: Model serialization

## Scalability Considerations

### Current Implementation
- Single patient assessment: ~2-5 seconds
- Batch processing: Sequential
- In-memory processing

### Future Enhancements
- Parallel batch processing
- GPU acceleration for embeddings
- Database integration for guidelines
- API deployment (REST/GraphQL)
- Real-time monitoring

## Safety & Ethics

### Built-in Safeguards
1. **Clear disclaimers**: Not a diagnostic tool
2. **Uncertainty quantification**: Explicit confidence intervals
3. **Evidence requirement**: All recommendations cited
4. **Limitation transparency**: What the system can't do
5. **Human oversight**: Designed for clinician review

### Privacy Considerations
- No data persistence by default
- HIPAA/GDPR compliance ready
- Anonymization support
- Audit trail capability

## Validation Strategy

### Model Validation
- Cross-validation (K-fold)
- Hold-out test set
- Calibration curves
- Multiple metrics (AUC, F1, Precision, Recall)

### Clinical Validation (Required)
- Expert review
- Clinical trials
- Comparison with standard care
- Regulatory approval

## Extension Points

### Easy to Extend
1. **Add new ML models**: Modify `agent1_risk_scoring.py`
2. **Add guidelines**: Edit `MEDICAL_GUIDELINES` in `config.py`
3. **Adjust risk thresholds**: Modify `ModelConfig` in `config.py`
4. **Add features**: Update data schema in `config.py`
5. **Custom summaries**: Extend `agent4_summary.py`

### Integration Points
- EHR systems (HL7/FHIR)
- Laboratory information systems
- Clinical decision support platforms
- Medical imaging systems
- Patient portals

## Performance Metrics

### ML Performance
- **AUC-ROC**: Model discrimination
- **Calibration**: Probability accuracy
- **F1-Score**: Balanced performance
- **Cross-validation**: Generalization

### Clinical Performance (to be validated)
- Diagnostic accuracy
- Time to appropriate care
- Patient outcomes
- Clinician satisfaction
- Resource utilization

---

This architecture provides a **solid foundation** for clinical decision support while maintaining **transparency, safety, and extensibility**.
