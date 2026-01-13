# ICU Deterioration Early-Warning System (MLOps)

## 📌 Project Overview

This project implements a **research-grade healthcare MLOps system** for predicting **ICU patient deterioration 4–6 hours in advance** using multivariate physiological time-series data.

The goal is twofold:

* **Academic signal (professors):** clinical relevance, rigorous modeling, explainability, and honest assumptions
* **Industry signal (recruiters):** end-to-end MLOps, reproducibility, deployment, monitoring, and system thinking

This is **not** a Kaggle-style or notebook-only project. It is designed as a realistic clinical decision-support *simulation*.

---

## 🧠 Clinical Motivation: What is ICU Deterioration?

In intensive care units, patient deterioration often manifests as **subtle physiological changes hours before critical events** (e.g., respiratory failure, sepsis, hemodynamic collapse).

Early-warning systems aim to:

* Continuously monitor vitals
* Detect abnormal temporal patterns
* Alert clinicians *before* irreversible damage occurs

This project mimics such systems by predicting deterioration risk using **recent time windows of vital signs**, not static snapshots.

---

## 📊 Dataset

**PhysioNet Challenge 2012**

* ~8,000 ICU stays
* Hourly multivariate time-series
* Realistic missingness and noise
* Open-access, clinically grounded

### Why this dataset?

* Represents real ICU workflows
* Avoids tabular shortcuts
* Enables temporal modeling and sliding-window prediction

---

## 🎯 Problem Framing

* **Input:** Past N-hour multivariate vital sign window
* **Output:** Binary risk of deterioration in the next 4–6 hours

### Why 4–6 hours?

* Clinically actionable window
* Matches real ICU escalation timelines
* Avoids trivial short-horizon predictions

---

## 📐 Modeling Strategy

### Architecture

* **1D CNN:** captures local temporal patterns (e.g., rapid SpO₂ drops)
* **GRU:** models longer-term physiological trends

Baselines:

* GRU-only
* CNN-only

This progression establishes scientific control before complexity.

---

## 📈 Evaluation Philosophy (Very Important)

ICU deterioration is a **highly imbalanced** problem.

### Why recall over accuracy?

* Missing a deteriorating patient is clinically costly
* Accuracy can be misleading when negatives dominate

### Target metric ranges (early-stage realism)

| Metric    | Target Range           |
| --------- | ---------------------- |
| Recall    | 0.85 – 0.92            |
| Precision | 0.25 – 0.45            |
| PR-AUC    | > baseline prevalence  |
| ROC-AUC   | ≥ 0.65                 |
| Accuracy  | ≥ 0.60 (not optimized) |

Dynamic thresholding and class-weighted loss were used to avoid trivial “predict-all-positive” behavior.

---

## ⚙️ MLOps Architecture

### High-level System Diagram

```
        ┌────────────┐
        │ Raw ICU    │
        │ Data       │
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ DVC        │  ← data & artifact versioning
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ Training   │  ← CNN+GRU (PyTorch)
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ MLflow     │  ← experiments & model registry
        └─────┬──────┘
              │
        ┌─────▼──────┐
        │ FastAPI    │  ← inference service
        └─────┬──────┘
              │
   ┌──────────▼──────────┐
   │ Prometheus / Grafana│  ← service monitoring
   └─────────────────────┘
```

---

## 📦 Data & Experiment Management

### DVC

* Tracks raw data, processed features, and model artifacts
* Enables reproducibility across experiments

Example:

```bash
dvc repro
```

### MLflow (via DagsHub)

* Logs parameters, metrics, and artifacts
* Model promotion based on metric thresholds

---

## 🔁 Pipeline Orchestration (Airflow)

Airflow orchestrates the **existing pipeline**, not the ML logic.

DAG stages:

1. Data ingestion
2. Validation
3. Feature generation
4. Training
5. Evaluation
6. Conditional model registration

Airflow schedules *when* things run; DVC defines *what* runs.

---

## 🚀 Serving & Deployment

### Backend (FastAPI)

Endpoints:

* `/health`
* `/predict`
* `/metrics`

The API loads the **Production model dynamically from MLflow**, enabling model updates without rebuilding images.

### Frontend

* Simulated ICU patient timeline
* Time slider + vital sign trends
* Risk score and alert level

This avoids unrealistic manual feature entry and mirrors ICU monitoring behavior.

---

## 📊 Monitoring

### Service Monitoring (Prometheus + Grafana)

Tracks:

* Request count
* Latency
* Error rate
* Risk-score distribution

This answers: *“Is the system behaving safely in real time?”*

---

## 📉 Data Drift Detection (Evidently AI)

No live data stream is available.

### Approach

* **Reference:** earlier ICU cohort
* **Current:** later ICU cohort

Evidently is used to **simulate post-deployment drift** via temporal splits.

Output:

* HTML drift report

This demonstrates capability without overstating production claims.

---

## 🧪 CI/CD

### CI (GitHub Actions)

* Linting
* Unit tests
* DAG import validation
* Docker image builds

CI is code-driven.

### Model updates

Handled via **MLflow registry**, not CI/CD pipelines.

---

## 🧭 Design Philosophy

* Prefer correctness over complexity
* Separate training, serving, and orchestration concerns
* Document limitations explicitly
* Avoid pretending to have live ICU data

---

## 📌 Project Status

✔ Completed end-to-end lifecycle
✔ Reproducible
✔ Honest about constraints

Future extensions are possible but intentionally deferred.

---

## 👤 Author

**Santosh Sapkota**
Healthcare ML & MLOps
