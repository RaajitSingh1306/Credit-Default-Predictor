# Credit Default Predictor

An end-to-end machine learning project that predicts the probability of loan default from applicant and loan information. The project includes model training, walk-forward validation, SHAP-based explainability, a FastAPI REST API, and a Streamlit web interface.

---

## What This Project Does

Given a loan applicant's details, the system predicts:

* **Default probability** — probability between 0 and 1
* **Prediction label** — `Approved` or `Default`
* **SHAP explanations** — identifies which features contributed most to the prediction

> **Note:** The `Approved` label means the model predicts that the applicant will **not default**. It is not an actual credit-approval decision.

---

## Project Structure

```text
credit-default-predictor/
├── main.ipynb           # ML pipeline: EDA → preprocessing → modelling → SHAP
├── app.py               # FastAPI backend
├── streamlit_app.py     # Streamlit frontend
├── model.pkl            # Trained Random Forest model
├── loan_data.csv        # Loan dataset
└── README.md
```

---

# Machine Learning Pipeline

The complete modelling workflow is contained in `main.ipynb`.

## 1. Exploratory Data Analysis & Cleaning

The dataset contains approximately **45,000 loan applications**.

The preprocessing workflow includes:

* Removing impossible ages (`age > 100`)
* Handling employment-experience outliers
* Capping income at the **99th percentile** to reduce the influence of extreme values while retaining observations
* Examining feature distributions
* Comparing default rates across categorical variables
* Analysing correlations between numerical features

---

## 2. Feature Encoding

Different encoding strategies are used depending on the nature of each variable.

### Binary Features

Binary categorical variables such as gender and previous loan defaults are encoded as `0/1`.

### Ordinal Features

Education is encoded according to its natural order:

```text
High School < Associate < Bachelor < Master < Doctorate
```

### Nominal Features

Variables without an inherent order, such as:

* Home ownership
* Loan intent

are one-hot encoded.

`drop_first=True` is used to remove the redundant reference category.

---

## 3. Modelling

### Baseline — Decision Tree

A shallow Decision Tree (`max_depth=5`) is used as an interpretable baseline.

It helps establish a simple benchmark and provides an intuitive view of how individual feature splits affect predictions.

### Final Model — Random Forest

The final model is a **Random Forest classifier** with:

* 100 trees
* `class_weight='balanced'`

The dataset contains approximately **22% default cases**, creating class imbalance.

Using `class_weight='balanced'` gives greater importance to the minority class during training and helps prevent the model from favouring the majority class.

---

## 4. Walk-Forward Cross-Validation

Instead of randomly splitting the observations, the project uses **`TimeSeriesSplit` with 5 folds**.

The expanding-window approach follows the pattern:

```text
Fold 1:
Train → rows 0–9k
Test  → rows 9k–18k

Fold 2:
Train → rows 0–18k
Test  → rows 18k–27k

...

Fold 5:
Train → earlier observations
Test  → later observations
```

This approach is intended to simulate a deployment scenario where a model is trained using historical applications and then used to predict applications arriving later.

Random cross-validation can produce overly optimistic results when observations have a meaningful temporal ordering because future observations can influence model training.

> **Important:** `TimeSeriesSplit` is appropriate only if the dataset is actually ordered chronologically. If `loan_data.csv` is not time-ordered, this validation strategy should not be interpreted as true temporal validation.

---

## 5. SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** with `TreeExplainer` to explain Random Forest predictions.

SHAP is used at two levels:

### Global Explainability

A SHAP summary plot shows which features are generally the most influential across the dataset.

### Individual Predictions

For each applicant, SHAP values show how individual features contributed to the model's prediction.

For example:

```text
loan_percent_income     +0.12
credit_score            -0.08
loan_int_rate           +0.06
```

Positive contributions increase the model's estimated default risk, while negative contributions reduce it.

---

# API

The backend is built using **FastAPI** and **Pydantic**.

## `POST /predict`

The endpoint accepts the applicant's input features as JSON and returns the model prediction, default probability, and SHAP explanations.

Example response:

```json
{
  "prediction": 1,
  "label": "Default",
  "default_probability": 0.73,
  "shap_values": {
    "loan_percent_income": 0.12,
    "credit_score": -0.08,
    "loan_int_rate": 0.06
  },
  "top_drivers": [
    {
      "feature": "loan_percent_income",
      "shap": 0.12
    },
    {
      "feature": "credit_score",
      "shap": -0.08
    }
  ]
}
```

## Start the API

```bash
uvicorn app:app --reload
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# Streamlit UI

The frontend is built with **Streamlit**.

The interface provides:

* Input fields for the applicant's features
* Default-risk prediction
* Default probability
* Top SHAP drivers
* Visual distinction between factors increasing and decreasing risk
* Expandable SHAP details

The Streamlit application communicates with the FastAPI backend through the `/predict` endpoint.

## Start the UI

Open a second terminal:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

# Running the Full Stack

### Terminal 1 — FastAPI

```bash
uvicorn app:app --reload
```

### Terminal 2 — Streamlit

```bash
streamlit run streamlit_app.py
```

The application will then be available at:

```text
http://localhost:8501
```

---

# Key Concepts

| Concept          | Where Used          | Purpose                                          |
| ---------------- | ------------------- | ------------------------------------------------ |
| Outlier handling | EDA / preprocessing | Reduce the influence of problematic observations |
| Ordinal encoding | Education           | Preserve meaningful category ordering            |
| One-hot encoding | Nominal categories  | Represent unordered categorical variables        |
| Class weighting  | Random Forest       | Handle class imbalance                           |
| Walk-forward CV  | Model validation    | Evaluate performance on later observations       |
| SHAP             | Explainability      | Explain global and individual predictions        |
| FastAPI          | Deployment          | Serve predictions through a REST API             |
| Pydantic         | API validation      | Validate incoming request data                   |
| Streamlit        | Frontend            | Provide an interactive prediction interface      |

---

# Results

Using the current pipeline, the model achieves approximately:

**ROC-AUC: 0.93–0.95 across walk-forward validation folds**

The strongest predictive features include:

1. `loan_percent_income`
2. `credit_score`
3. `previous_loan_defaults_on_file`
4. `loan_int_rate`
5. `loan_amnt`

Performance should be interpreted alongside the individual fold results and the exact train/test setup used in the notebook.

---

# Tech Stack

* **Python**
* **Pandas / NumPy** — data processing
* **Scikit-learn** — preprocessing, modelling, and validation
* **SHAP** — model explainability
* **FastAPI** — REST API
* **Pydantic** — request validation
* **Streamlit** — interactive UI
* **Jupyter Notebook** — experimentation and model development

---

# Disclaimer

This project is intended for **educational and demonstration purposes**.

A machine-learning prediction of default risk should not be treated as a standalone lending or credit-approval decision. Real-world credit systems require additional considerations including regulatory requirements, fairness and bias evaluation, data quality, model monitoring, calibration, security, and human oversight.
