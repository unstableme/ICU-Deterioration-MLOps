# 🏥 ICU Patient Deterioration Early-Warning System

**Healthcare based complete MLOps project** that predicts whether an ICU patient is likely to deteriorate in the **next 6 hours** using multivariate time-series vital signs.

This project is intentionally designed to demonstrate **clinical ML reasoning**, **end-to-end MLOps discipline**, and **production awareness**, rather than just model accuracy.

** Visit the web here:** https://icu-deterioration-frontend-production.up.railway.app/

---

## 🎯 Project Motivation

Early detection of patient deterioration in Intensive Care Units (ICUs) is critical. Delayed intervention can significantly increase mortality, length of stay, and healthcare costs.

This system mimics real-world clinical decision-support systems by:

* Operating on **time-series physiological signals**
* Emphasizing **recall (sensitivity)** over raw accuracy
* Treating ML as a **continuously monitored system**, not a static notebook

---

## 🧠 Problem Formulation

**Task**: Binary classification
**Goal**: Predict whether a patient will deteriorate within the next **6 hours**

**Positive class (1)**: Patient deteriorates
**Negative class (0)**: Patient remains stable

### Why Recall > Accuracy

In clinical settings:

* **False Negatives** (missing a deteriorating patient) are far more dangerous than false positives
* A false alert can be reviewed by clinicians
* A missed alert can cost a life

Therefore:

* **Recall** is prioritized during model evaluation
* Accuracy alone is considered misleading for this problem

---

## 📊 Dataset

* ICU time-series vitals (PhysioNet-style structure) of total 12k data only around 15% were from positive class (highly imbalanced dataset).
* Features include:

  * Heart Rate
  * Blood Pressure (Systolic / Diastolic)
  * Respiratory Rate
  * SpO₂
  * Other physiological indicators
  
* Set A is used for training, Set B is used for validation, and Set C is used for testing.  


## 🧠 Model Architecture

* **1D CNN** for local temporal pattern extraction
* **GRU (Gated Recurrent Unit)** for sequential dependency modeling

This hybrid design allows:

* CNN → short-term signal patterns
* GRU → longer temporal dependencies

The architecture balances **performance** and **computational feasibility**.

---

## ⚙️ System Architecture (End-to-End)

```
                ┌────────────────────────────┐
                │      Raw ICU Data          │
                │   (PhysioNet-style)        │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │   Data Processing Pipeline │
                │  (Cleaning, Windowing,     │
                │   Feature Engineering)     │
                └─────────────┬──────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
┌──────────────────────┐               ┌──────────────────────┐
│  DVC (Data & Artifact│               │ Evidently AI (Drift  │
│  Versioning)         │               │ Analysis)            │
│  - raw data          │               │ Reference vs Current │
│  - processed data    │               │ HTML Report          │
└───────────┬──────────┘               └──────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────┐
│          Training & Evaluation Pipeline      │
│      (CNN + GRU, PyTorch)                    │
│                                              │
│  Metrics: Recall, Precision, PR-AUC, ROC-AUC │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│              MLflow                          │
│  - Experiment Tracking                       │
│  - Metrics & Artifacts                       │
│  - Model Registry (Conditional Promotion)    │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│              Airflow                         │
│  Orchestrates:                               │
│  - DVC Repro                                 │
│  - Training                                  │
│  - Evaluation                                │
│  - Registry Decision                         │
└───────────────┬──────────────────────────────┘
                │
                ▼
        ┌────────────────────────────┐
        │   Docker Images Built via  │
        │   CI (GitHub Actions)      │
        │                            │
        │  - ML Training Image       │
        │  - FastAPI Backend Image   │
        │  - Frontend Image          │
        └─────────────┬──────────────┘
                      │
                      ▼
              ┌──────────────────┐
              │   Docker Hub     │
              └───────┬──────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
┌──────────────────────┐    ┌──────────────────┐
│   ML Inference Image │    │ FastAPI Backend  │
│   (Loaded from       │───►│  /predict        │
│    MLflow Registry)  │    │  /metrics        │
└──────────────────────┘    └──────────┬───────┘
                                       │
                        ┌──────────────┴───────────────┐
                        ▼                              ▼
            ┌────────────────────────────┐   ┌──────────────────────────────────────────────┐
            │        Frontend UI         │   │     Prometheus → Grafana                     │
            │  (Risk Score Visualization)│   │  - Request Rate                              │
            └────────────────────────────┘   │  - Latency                                   │
                                             │  - Error Rate                                │
                                             │  - Risk Score Distribution                   │
                                             └──────────────────────────────────────────────┘
```
---


## 🧪 Training & Experiment Tracking

### MLflow

Used for:
- Experiment tracking
- Metric logging (Recall, Precision, Loss)
- Model versioning

**Important Design Choice**:
- Models are **registered only if they meet a recall threshold**
- This enforces clinical safety logic directly in the pipeline

---

## 🛠️ Orchestration with Airflow

Apache Airflow is used to orchestrate the ML pipeline:

- Data preparation
- Model training
- Evaluation
- Conditional model registration

Runs can be:
- Triggered manually (current setup)
- Scheduled (e.g., daily retraining)

---

## 🚀 Inference API (FastAPI)

A lightweight **FastAPI** service provides:
- `/predict` → Risk score inference
- `/metrics` → Prometheus-compatible metrics

---

## 📈 Monitoring (Prometheus + Grafana)

### Metrics Tracked

- Request count
- Request latency
- Error rate
- Risk score distribution (histogram)

Prometheus acts as a **metrics collector**, Grafana as the **visual layer**.

---

## 🔍 Data Drift Detection (Evidently AI)

Even without continuous data flow, drift detection was implemented to demonstrate **production awareness**.

### Approach

- Reference dataset vs current dataset (Set A vs Set C)
- Statistical drift detection
- HTML report generation

### Outcome

- ~2% drift detected
- Confirms data stability in controlled setup

This step shows readiness for **real-world deployment**, where drift is unavoidable.

---

## 🐳 Containerization

- Docker used for:
  - Training pipeline
  - Inference service
  - Monitoring stack

---

## 🔁 CI/CD (GitHub Actions)

### Continuous Integration (CI)

- Code linting
- Docker build validation
- Pipeline consistency checks

### Continuous Deployment (CD)

- API redeployment on main branch updates
- Model lifecycle handled via MLflow (not CI)

---

## 🧠 Final Note

This project represents **how ML systems behave in the real world** — imperfect data, safety constraints, monitoring, and accountability.

It is intentionally scoped to demonstrate **depth over breadth**.

```
