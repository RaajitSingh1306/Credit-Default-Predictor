# Credit Default Predictor — End-to-End ML & Explainability Pipeline

[![FastAPI Serving](https://img.shields.io/badge/FastAPI-REST%20API-009688)](#fastapi-backend-api)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive%20Frontend-red)](#streamlit-web-interface)
[![Model Validation](https://img.shields.io/badge/ROC--AUC-0.93--0.95-emerald)](#validation--model-performance)
[![SHAP Explainability](https://img.shields.io/badge/Explainability-Tree%20SHAP-blue)](#5-explainability-shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning system that predicts the probability of loan default from applicant demographics, financial ratios, and credit bureau histories. The project features full exploratory data analysis (EDA), rigorous class-imbalanced Random Forest modeling, walk-forward validation (ROC-AUC 0.93–0.95), exact TreeSHAP feature attribution, a **FastAPI REST API**, and a **Streamlit interactive lending dashboard**.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Why It Was Built](#why-it-was-built)
3. [Machine Learning Pipeline](#machine-learning-pipeline)
   - [1. Exploratory Data Analysis & Cleaning](#1-exploratory-data-analysis--cleaning)
   - [2. Feature Encoding & Engineering](#2-feature-encoding--engineering)
   - [3. Class Imbalance Management](#3-class-imbalance-management)
   - [4. Model Architecture & Walk-Forward Validation](#4-model-architecture--walk-forward-validation)
   - [5. Explainability (TreeSHAP)](#5-explainability-treeshap)
4. [Project Directory Layout](#project-directory-layout)
5. [Where & How to Start](#where--how-to-start)
   - [Step 1: Environment Setup](#step-1-environment-setup)
   - [Step 2: Train Model (Optional)](#step-2-train-model-optional)
   - [Step 3: Start FastAPI Backend](#step-3-start-fastapi-backend)
   - [Step 4: Start Streamlit Frontend](#step-4-start-streamlit-frontend)
6. [API Reference & Sample Request](#api-reference--sample-request)
7. [Validation & Model Performance](#validation--model-performance)
8. [Connected Portfolio Projects](#connected-portfolio-projects)
9. [Disclaimer](#disclaimer)

---

## What This Project Does

Given a prospective loan applicant's demographic, economic, and bureau data, the scoring engine delivers:

1. **Default Probability Score**: Precise probability between $0.00$ and $1.00$ of loan default.
2. **Actionable Classification**: Prediction label: `Approved` (Low Risk) or `Default` (High Risk).
3. **Local SHAP Feature Attribution**: Visual bar breakdown identifying exactly which applicant factors drove the score up or down (e.g., debt-to-income ratio, interest rate, credit history length).

> [!NOTE]
> The `Approved` label indicates the algorithmic assessment that the applicant is non-defaulting; it serves as a decision-support metric rather than an autonomous statutory approval.

---

## Why It Was Built

* **Transparent Credit Decisions**: Modern lending regulations (like RBI credit directions and FCRA) prohibit "black box" credit underwriting. Integrating SHAP directly into the serving layer guarantees that every rejection has an auditable, human-interpretable explanation.
* **Non-Linear Risk Profiling**: Linear scoring tables fail when interaction effects dominate (e.g., high income with extreme debt load or low credit history with high loan amounts). Random Forest captures deep variable interactions without overfitting.
* **Decoupled Production Architecture**: Separates the ML training pipeline (`main.ipynb`), low-latency inference backend (`app.py`), and loan officer user interface (`streamlit_app.py`).

---

## Architecture

```
main.ipynb (Training Pipeline)
    │
    ├── EDA & Boundary Cleaning (age bounds, income winsorization)
    ├── Feature Encoding (binary indicators, ordinal, one-hot)
    ├── Class Balancing (balanced sample weights)
    ├── Walk-Forward RF Training (n_estimators=200, max_depth=12)
    └── TreeSHAP Explainer Setup → models/model.pkl
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
app.py (FastAPI Backend)        streamlit_app.py (Loan Officer UI)
Port :8000                      Port :8501
- Pydantic schema validation    - Interactive parameter inputs
- Sub-25ms inference            - Score gauges & risk flags
- Exact TreeSHAP local values   - Live TreeSHAP contribution waterfall
```

---

## Machine Learning Pipeline

The complete pipeline is developed in `main.ipynb` over ~45,000 historical loan records:

### 1. Exploratory Data Analysis & Cleaning
* **Boundary Validation**: Eradicates spurious age inputs (`age > 100`) and employment duration anomalies.
* **Percentile Winsorization**: Capping income at the 99th percentile prevents outlier distortion while retaining genuine affluent borrowers.
* **Bivariate Correlation**: Analyzes feature correlations and default rate distributions across categorical buckets.

### 2. Feature Encoding & Engineering
* **Binary Variables**: `person_gender` and `previous_loan_defaults_on_file` mapped to $0/1$.
* **Ordinal Variables**: `person_education` sequentially ranked (`High School`: 0 $\rightarrow$ `Doctorate`: 4).
* **Nominal Features**: `person_home_ownership` (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) and `loan_intent` (`EDUCATION`, `MEDICAL`, `VENTURE`, etc.) one-hot encoded.
* **Interaction Ratio**: `loan_percent_income` computed as $\text{Loan Amount} / \text{Annual Income}$.

### 3. Class Imbalance Management
Credit defaults are naturally sparse (~20% default rate in raw data). Mitigated through balanced class weighting:
$$\text{Weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$

### 4. Model Architecture & Walk-Forward Validation
* **Model**: Scikit-Learn `RandomForestClassifier` with tuned hyper-parameters (`n_estimators=200`, `max_depth=12`, `class_weight='balanced'`).
* **Validation**: Walk-forward expanding time splits to evaluate robustness on future loan cohorts rather than random cross-validation.

### 5. Explainability (TreeSHAP)
* Employs `shap.TreeExplainer` directly on the fitted forest.
* Pre-computes background expected values at server startup to enable real-time ($<25\text{ms}$) per-applicant Shapley decomposition.

---

## Project Directory Layout

```text
Credit Default Predictor/
├── main.ipynb              # ML pipeline: EDA → feature engineering → RF training → SHAP
├── app.py                  # FastAPI REST backend serving predictions & SHAP vectors
├── streamlit_app.py        # Streamlit lending officer dashboard
├── model.pkl               # Serialized Random Forest model bundle
├── loan_data(best).csv     # Preprocessed loan dataset (~45k records)
├── requirements.txt        # Dependencies (scikit-learn, fastapi, streamlit, shap, etc.)
└── README.md               # Project documentation
```

---

## Where & How to Start

### Step 1: Environment Setup

```bash
cd "Credit Default Predictor"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train Model (Optional)

A high-performing pre-trained model is already serialized as `model.pkl`. To re-run EDA, modify features, or retrain:

```bash
jupyter notebook main.ipynb
```

### Step 3: Start FastAPI Backend

Launch the REST inference server in **Terminal 1**:

```bash
uvicorn app:app --reload --port 8000
```

* API Base URL: **`http://127.0.0.1:8000`**
* Interactive Swagger Docs: **`http://127.0.0.1:8000/docs`**

### Step 4: Start Streamlit Frontend

In **Terminal 2**, launch the loan officer interface:

```bash
streamlit run streamlit_app.py --server.port 8501
```

* Web UI URL: **`http://localhost:8501`**

---

## API Reference & Sample Request

### Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Health check and service status |
| `POST` | `/predict` | Evaluates loan application and returns default probability + SHAP |

### Sample JSON Request Payload (`POST /predict`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 28,
    "person_gender": 1,
    "person_education": 2.0,
    "person_income": 45000,
    "loan_amnt": 12000,
    "loan_int_rate": 14.5,
    "loan_percent_income": 0.2667,
    "cb_person_cred_hist_length": 4,
    "credit_score": 620,
    "previous_loan_defaults_on_file": 0,
    "person_home_ownership_OTHER": 0,
    "person_home_ownership_OWN": 0,
    "person_home_ownership_RENT": 1,
    "loan_intent_EDUCATION": 0,
    "loan_intent_HOMEIMPROVEMENT": 0,
    "loan_intent_MEDICAL": 0,
    "loan_intent_PERSONAL": 0,
    "loan_intent_VENTURE": 0
  }'
```

### Sample JSON Response

```json
{
  "prediction": "Approved",
  "default_probability": 0.184,
  "top_risk_factors": [
    {"feature": "loan_percent_income", "shap_value": 0.082},
    {"feature": "loan_int_rate", "shap_value": 0.045},
    {"feature": "credit_score", "shap_value": -0.112}
  ]
}
```

---

## Validation & Model Performance

* **Cross-Validation Scheme**: Expanding-window walk-forward validation across chronological cohorts.
* **ROC-AUC**: **0.934 – 0.951** across all validation folds.
* **Top Predictive Drivers**:
  1. `loan_percent_income` (Loan as percentage of annual income)
  2. `credit_score` (Credit bureau score)
  3. `previous_loan_defaults_on_file` (Historical delinquency flag)
  4. `loan_int_rate` (Assigned loan interest rate)
  5. `loan_amnt` (Principal loan amount)

---

## Key Design Decisions

- **Random Forest over Gradient Boosting**: Random Forest with `class_weight='balanced'` handles imbalanced credit defaults robustly without requiring synthetic sampling (SMOTE) or aggressive focal loss tuning, while providing monotonic tree ensembles favored by regulatory credit risk auditors.
- **Expanding-Window Walk-Forward Validation over K-Fold**: Loan defaults are chronologically clustered; standard random k-fold cross-validation leaks forward-looking macro conditions into historical folds, producing artificially inflated ROC-AUC.
- **TreeSHAP over KernelSHAP**: Exact polynomial-time Shapley computation for tree ensembles via TreeSHAP allows real-time local attribution (<25ms per applicant) directly inside the API request-response cycle.
- **Pre-Computed SHAP Background at Startup**: `app.py` loads `model.pkl` and initializes the TreeSHAP explainer once at server startup, avoiding per-request background re-sampling overhead.
- **Decoupled Stateless Architecture**: `main.ipynb` performs heavy analytical exploration and model persistence; `app.py` remains a lightweight stateless microservice; `streamlit_app.py` serves as the loan officer frontend.

---

## Connected Portfolio Projects

* **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: AI assistant covering RBI Model Risk Management directions and digital personal data compliance.
* **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: TreeSHAP explainability utilized for macroeconomic volatility regime classification.

---

## Limitations & Roadmap

### Known Limitations
- **Static Ingestion**: The training dataset (`loan_data(best).csv`, ~45,000 observations) represents a historical static cohort with no real-time loan origination system (LOS) streaming feed.
- **Absence of Fair Lending / Disparate Impact Audit**: Does not implement formal algorithmic fairness audits (demographic parity, equalized odds) across protected demographic classes.
- **Self-Reported Income**: Relies on self-reported `person_income` without automated payroll/tax bank verification APIs.
- **Single Model Family**: Production serving utilizes Random Forest only; no multi-model dynamic routing or stacking with LightGBM/CatBoost.
- **No Population Stability Monitoring**: Lacks automated tracking for Population Stability Index (PSI) or Characteristic Stability Index (CSI) to detect post-deployment credit score drift.

### Roadmap
- [ ] **Multi-Model Benchmark Suite**: Add automated hyperparameter tuning and benchmarking against LightGBM, CatBoost, and TabNet.
- [ ] **Fairness & Bias Audit Engine**: Implement AIF360 / Fairlearn metrics to guarantee regulatory compliance with Fair Lending standards.
- [ ] **Drift & PSI Monitoring**: Implement an automated monitoring dashboard for feature distribution shifts and Population Stability Index.
- [ ] **Live Open Banking Ingestion**: Integrate Account Aggregator (AA) sandbox APIs for automated bank statement analysis.

---

## Disclaimer

This system is built for educational, research, and portfolio demonstration purposes. Real-world credit underwriting requires regulatory compliance (fair lending audits, disparate impact analysis), model monitoring for covariate shift, and certified human-in-the-loop validation.
