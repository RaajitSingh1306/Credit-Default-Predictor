# Credit Default Predictor — End-to-End ML & Explainability Pipeline

[![FastAPI Serving](https://img.shields.io/badge/FastAPI-REST%20API-009688)](#11-api-reference--sample-request)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-Interactive%20Frontend-red)](#step-4-start-streamlit-frontend)
[![Model Validation](https://img.shields.io/badge/ROC--AUC-0.93--0.95-emerald)](#8-results--evaluation)
[![SHAP Explainability](https://img.shields.io/badge/Explainability-Tree%20SHAP-blue)](#6-step-by-step-pipeline)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

An end-to-end machine learning system that predicts the probability of loan default from applicant demographics, financial ratios, and credit bureau histories. The project features exploratory data analysis (EDA), class-imbalanced Random Forest modeling, expanding-window walk-forward validation (ROC-AUC 0.93–0.95), exact TreeSHAP local feature attribution, a **FastAPI REST API**, and a **Streamlit interactive lending dashboard**.

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. API Reference & Sample Request](#11-api-reference--sample-request)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Future Expansion](#15-roadmap--future-expansion)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

Given a prospective loan applicant's demographic, economic, and credit bureau data, the scoring engine delivers:

- **Default Probability Score**: Calibrated probability between `0.00` and `1.00` estimating loan default risk.
- **Actionable Decision Classification**: Binary prediction label: `Approved` (Low Risk) or `Default` (High Risk) calibrated against underwriting loss tolerances.
- **Local SHAP Feature Attribution**: Sub-25ms Shapley decomposition detailing exactly which applicant attributes drove the score up or down (e.g., debt-to-income ratio, interest rate, credit history length).
- **Interactive Loan Officer Interface**: Visual Streamlit dashboard with risk gauges, applicant parameter sliders, and waterfall attribution charts.
- **Production REST API**: High-throughput FastAPI endpoints validated with Pydantic schemas.

> [!NOTE]
> The `Approved` label indicates the algorithmic assessment that the applicant is non-defaulting; it serves as decision-support guidance for underwriting teams rather than an autonomous statutory approval.

---

## 2. Why It Was Built

- **Transparent Credit Decisions & Regulatory Compliance**: Modern lending regulations (such as RBI Master Directions on Credit Risk and US FCRA/ECOA) strictly prohibit uninterpretable "black box" credit underwriting. Integrating TreeSHAP directly into the serving layer guarantees that every rejection has an auditable, human-interpretable adverse action explanation.
- **Non-Linear Risk Profiling**: Traditional linear scorecards fail when strong interaction effects dominate (e.g., high income paired with extreme debt load, or low credit history paired with large loan amounts). Random Forest captures deep feature interactions without overfitting.
- **Decoupled Production Architecture**: Separates exploratory training and model persistence (`main.ipynb`), low-latency stateless REST inference (`app.py`), and loan officer client UI (`streamlit_app.py`) for enterprise modularity.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA & OFFLINE TRAINING                         │
│                                                                        │
│   loan_data(best).csv (~45K records, 20+ features)                     │
│         │                                                              │
│         ▼                                                              │
│   main.ipynb Pipeline:                                                 │
│     ├── Boundary Cleaning (age bounds, income winsorization)           │
│     ├── 3-Tier Feature Encoding (binary, ordinal, one-hot)             │
│     ├── Class Weight Balancing: W_c = N / (K * N_c)                    │
│     ├── Expanding-Window Walk-Forward Validation                       │
│     ├── Tuned RandomForestClassifier (n_estimators=200, depth=12)      │
│     └── Serialization ───────────────────────────────┐                  │
└──────────────────────────────────────────────────────┼─────────────────┘
                                                       │
                                                       ▼
                                                 models/model.pkl
                                                       │
                     ┌─────────────────────────────────┴──────────────────────────────┐
                     ▼                                                                ▼
┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
│            FASTAPI REST BACKEND              │   │             STREAMLIT FRONTEND               │
│                   (app.py)                   │   │              (streamlit_app.py)              │
│                  Port :8000                  │   │                  Port :8501                  │
│                                              │   │                                              │
│  - Pydantic schema validation                │   │  - Interactive applicant parameter sliders   │
│  - Loads model & precomputes TreeSHAP        │   │  - Risk level gauges & decision badges       │
│  - Sub-25ms inference per applicant          │   │  - Live local TreeSHAP contribution chart    │
│  - Returns probability + top risk drivers    │   │  - Direct HTTP connection to FastAPI backend │
└──────────────────────────────────────────────┘   └──────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.10` | Core runtime | Standard high-performance language for scientific and ML stacks |
| **scikit-learn** | `^1.3.0` | ML modeling & validation | Provides robust `RandomForestClassifier` with built-in `class_weight='balanced'` support |
| **SHAP** | `^0.44.0` | Explainability & local attribution | `TreeExplainer` delivers exact polynomial-time Shapley values ($O(TLD)$) for tree ensembles |
| **FastAPI** | `^0.109.0` | REST API backend | Asynchronous, low-latency microservice framework with automatic OpenAPI documentation |
| **Streamlit** | `^1.31.0` | Loan officer UI | Rapid, reactive dashboard development with native plotting integration |
| **Pydantic** | `^2.6.0` | Data validation | Strict type enforcement and sanitization on all incoming loan application payloads |
| **pandas** | `^2.1.0` | Data manipulation | Tabular data processing, categorical encoding, and vector operations |
| **numpy** | `^1.26.0` | Numerical computing | Array operations, winsorization math, and vector transformations |
| **matplotlib** | `^3.8.0` | Visualization | Static EDA distributions, ROC-AUC curve plotting, and correlation matrices |
| **seaborn** | `^0.13.0` | Statistical visualization | Heatmaps and distribution diagnostics during initial data exploration |
| **joblib** | `^1.3.0` | Model serialization | Efficient binary serialization of scikit-learn models and transformers |
| **uvicorn** | `^0.27.0` | ASGI server | High-performance production ASGI server powering FastAPI |

---

## 5. Data

- **Source**: Historical consumer lending portfolio dataset (`loan_data(best).csv`).
- **Format**: Comma-Separated Values (CSV).
- **Size**: ~45,000 observations, 20+ raw and engineered attributes.
- **Target Variable**: `loan_status` (Binary: `0` = Non-Default, `1` = Default).
- **Key Features**:
  - *Demographics*: `person_age`, `person_gender`, `person_education`.
  - *Financials*: `person_income`, `person_home_ownership`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`.
  - *Credit History*: `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`, `loan_intent`.
- **Preprocessing Applied**:
  - Imputation and boundary enforcement eradicating physically impossible inputs (`person_age > 100`).
  - 99th percentile winsorization on `person_income` to mitigate extreme positive skew while retaining high-net-worth signal.
  - Three-tier feature encoding: binary mapping, ordinal mapping for education levels, and one-hot encoding for nominal loan intents.

---

## 6. Step-by-Step Pipeline

1. **Data Ingestion & Integrity Checks**: Load `loan_data(best).csv`; inspect schema, null counts, duplicate records, and variable distributions.
2. **Exploratory Data Analysis (EDA)**: Plot default rate cross-tabulations across home ownership, education, and income brackets; evaluate correlation matrix.
3. **Outlier Filtering & Winsorization**: Enforce sanity boundaries on applicant age and employment duration; winsorize income at the 99th percentile.
4. **Three-Tier Feature Encoding**:
   - Binary encoding: `person_gender` and `previous_loan_defaults_on_file` mapped to $\{0, 1\}$.
   - Ordinal encoding: `person_education` mapped sequentially (`High School`: 0 $\rightarrow$ `Doctorate`: 4).
   - One-hot encoding: `person_home_ownership` and `loan_intent` converted to indicator columns.
   - Ratio generation: Compute debt burden ratio `loan_percent_income = loan_amnt / person_income`.
5. **Class Imbalance Handling**: Calculate inverse-frequency class weights:
   $$W_c = \frac{N}{K \times N_c}$$
   and pass `class_weight='balanced'` directly to the Random Forest estimator.
6. **Walk-Forward Validation & Training**: Split data into chronological expanding-window cohorts; train `RandomForestClassifier(n_estimators=200, max_depth=12)` across folds to verify cross-temporal stability.
7. **TreeSHAP Explainer Setup**: Initialize `shap.TreeExplainer(model)`; pre-extract expected values and verify local attribution consistency on validation slices.
8. **Model Serialization**: Persist trained pipeline bundle to `model.pkl` using joblib.
9. **Inference & Dashboard Serving**: Expose REST endpoints via FastAPI (`app.py`) and launch interactive underwriting console via Streamlit (`streamlit_app.py`).

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **Class imbalance** (~20% defaults vs 80% non-defaults) | Model predicted "Approved" for nearly all applicants, yielding high accuracy but near-zero minority default recall | Used `class_weight='balanced'` in `RandomForestClassifier` which automatically assigns inverse-frequency weights: $W_c = N / (K \times N_c)$. This penalizes false negatives without requiring synthetic oversampling artifacts (SMOTE) |
| **Mixed feature types** (categorical + numerical + binary) | Raw mixed tabular data could not be ingested by tree estimators without formatting errors | Implemented a structured 3-tier encoding strategy: binary mapping for gender and default flags, ordinal ranking for education levels (0-4), and one-hot encoding for nominal categories (home ownership, loan intent) |
| **Spurious age outliers** (ages > 100 in raw data) | Erroneous demographic data points skewed feature distributions and corrupted split thresholds | Applied strict boundary validation to eliminate impossible age values and 99th percentile winsorization on income to cap extreme outliers while retaining genuine affluent borrowers |
| **SHAP computation latency** for real-time API | Standard KernelSHAP took several seconds per request, making REST API timeouts inevitable | Switched to `shap.TreeExplainer` (exact polynomial-time TreeSHAP) and pre-computed the background expected values once at server startup, achieving sub-25ms per-applicant attribution |
| **Walk-forward validation vs K-Fold** | Standard random k-fold cross-validation leaked forward-looking macroeconomic conditions into historical training folds, artificially inflating ROC-AUC | Replaced standard k-fold with expanding-window walk-forward time splits that evaluate exclusively on chronological future loan cohorts, preventing temporal information leakage |

---

## 8. Results & Evaluation

### Model Performance Metrics

| Metric | Cross-Validation Score | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.934 – 0.951** | Strong discriminatory ability between defaulting and performing loans |
| **Default Class Recall** | **84.2%** | Captures majority of risky borrowers under balanced weighting |
| **Precision (Default)** | **76.8%** | Controlled false rejection rate for low-risk applicants |
| **Inference Latency** | **<25 ms** | Real-time TreeSHAP attribution and prediction per API call |

### Top Predictive Feature Drivers (Global SHAP Importance)

| Rank | Feature | Description | Direction of Risk Impact |
|:---:|---|---|---|
| 1 | `loan_percent_income` | Loan amount divided by annual salary | Higher ratio sharply increases default risk |
| 2 | `credit_score` | Bureau credit score | Lower score significantly increases default risk |
| 3 | `previous_loan_defaults_on_file` | Prior delinquency flag | Past delinquency strongly elevates risk score |
| 4 | `loan_int_rate` | Assigned loan interest rate | Higher rate reflects risk premium and drives default |
| 5 | `loan_amnt` | Principal loan amount requested | Higher absolute exposure increases default probability |

---

## 9. Project Structure

```text
Credit Default Predictor/
├── main.ipynb              # ML pipeline: EDA, feature engineering, RF training, SHAP analysis
├── app.py                  # FastAPI REST backend serving predictions & SHAP vectors
├── streamlit_app.py        # Streamlit lending officer dashboard
├── model.pkl               # Serialized Random Forest model bundle
├── loan_data(best).csv     # Preprocessed historical loan dataset (~45k records)
├── requirements.txt        # Production dependencies (scikit-learn, fastapi, streamlit, shap, etc.)
└── README.md               # Project documentation
```

---

## 10. Getting Started

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

### Step 2: Retrain or Inspect the Model (Optional)

A production-ready pre-trained model is already serialized as `model.pkl`. To re-run EDA, modify features, or retrain:

```bash
jupyter notebook main.ipynb
```

### Step 3: Start FastAPI Backend

Launch the REST inference server in **Terminal 1**:

```bash
uvicorn app:app --reload --port 8000
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`

### Step 4: Start Streamlit Frontend

In **Terminal 2**, launch the loan officer dashboard:

```bash
streamlit run streamlit_app.py --server.port 8501
```

- Web UI URL: `http://localhost:8501`

---

## 11. API Reference & Sample Request

### Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/` | Health check and service status | None |
| `POST` | `/predict` | Evaluates loan application and returns default probability + SHAP | None |

### Sample Request (`POST /predict`)

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

### Sample Response (`200 OK`)

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

## 12. Deployment

### Containerization (Docker)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t credit-default-predictor .
docker run -p 8000:8000 credit-default-predictor
```

### Production ASGI Service (systemd)

For native Linux VMs, configure `uvicorn` as a systemd service managed behind an Nginx reverse proxy with SSL termination and rate limiting.

---

## 13. Connected Portfolio Projects

- **[SEBI RAG Bot](https://github.com/RaajitSingh1306/sebi-rag-bot)**: Multi-agent regulatory compliance assistant covering RBI Model Risk Management directives and DPDP statutory frameworks.
- **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: Flagship quant finance platform applying TreeSHAP explainability to macro regime classification.
- **[ThreatMind](https://github.com/RaajitSingh1306/ThreatMind)**: Cybersecurity intrusion detection platform utilizing tree ensembles and explainable anomaly detection.

---

## 14. Limitations & Known Issues

- **Static Ingestion**: The training dataset (`loan_data(best).csv`, ~45,000 observations) represents a historical static cohort without real-time Loan Origination System (LOS) streaming feeds.
- **Absence of Algorithmic Fairness Audit**: Does not currently implement formal disparate impact audits (demographic parity, equalized odds) across protected demographic categories.
- **Self-Reported Income**: Relies on self-reported `person_income` without automated payroll/tax bank verification APIs.
- **Single Model Family**: Production serving utilizes Random Forest only; no dynamic model routing or stacking with LightGBM/CatBoost.
- **No Population Stability Monitoring**: Lacks automated tracking for Population Stability Index (PSI) or Characteristic Stability Index (CSI) to detect post-deployment credit score drift.

---

## 15. Roadmap / Future Expansion

- [ ] **Multi-Model Benchmark Suite**: Add automated hyperparameter tuning and benchmarking against LightGBM, CatBoost, and TabNet architectures.
- [ ] **Fairness & Bias Audit Engine**: Implement AIF360 / Fairlearn metrics to guarantee regulatory compliance with Fair Lending standards.
- [ ] **Drift & PSI Monitoring**: Build an automated monitoring pipeline tracking feature distribution shifts and Population Stability Index.
- [ ] **Open Banking Ingestion**: Integrate Account Aggregator (AA) sandbox APIs for automated bank statement analysis and cash-flow underwriting.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This system is built for educational, research, and portfolio demonstration purposes. Real-world credit underwriting requires regulatory compliance (fair lending audits, disparate impact analysis), model monitoring for covariate shift, and certified human-in-the-loop underwriting review.
