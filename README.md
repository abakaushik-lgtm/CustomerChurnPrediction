# Customer Churn Prediction & Analytics Platform

> **Repository Description**: End-to-end Customer Churn Prediction & Analytics Platform using Python, Scikit-learn, SQL, Plotly, and Streamlit with interactive dashboards and machine learning insights.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Community%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://customer-churn-prediction-demo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![SQL](https://img.shields.io/badge/SQL-PostgreSQL%20%7C%20SQLite%20%7C%20DuckDB-blue.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Data%20Viz-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Demo

🚀 **Experience the Live Interactive Web App**: [https://customer-churn-prediction-demo.streamlit.app/](https://customer-churn-prediction-demo.streamlit.app/)

---

## 🏷️ Repository Topics

`python` • `machine-learning` • `customer-churn` • `classification` • `streamlit` • `plotly` • `sql` • `data-analysis` • `analytics` • `business-intelligence` • `scikit-learn`

---

## 📌 Project Overview

The **Customer Churn Prediction & Analytics Platform** is an enterprise-grade Data Science and Business Intelligence solution designed to detect customer attrition risk, analyze behavioral churn drivers, segment subscriber cohorts, and recommend automated retention playbooks for Customer Success teams.

---

## 💼 Business Problem

Acquiring new customers is **5x to 25x more expensive** than retaining existing ones. High monthly churn directly reduces Monthly Recurring Revenue (MRR) and compromises Customer Lifetime Value (CLV).

* **Baseline Churn Rate**: 26.5% (1,869 out of 7,043 customers)
* **Monthly Revenue at Risk**: $131,200 / month
* **Annual Revenue Impact**: $1.57 Million / year

### Strategic Objectives:
1. **Predictive Risk Scoring**: Classify subscribers into Low, Medium, or High Risk tiers before they churn.
2. **Behavioral Insight Discovery**: Uncover leading risk indicators (contract types, support tickets, monthly bills).
3. **Revenue Impact Simulation**: Model financial savings achieved by proactive retention campaigns.

---

## 📊 Dataset

The dataset consists of **7,043 customer accounts** evaluated across **20 feature attributes**:

* **Demographics**: Gender, Senior Citizen, Partner, Dependents
* **Account Info**: Tenure (months), Contract Type (Monthly, Annual), Subscription Tier (Basic, Premium), Paperless Billing, Payment Method
* **Usage & Engagement**: Monthly Charges, Total Charges, Support Tickets, Login Frequency, Usage Hours
* **Service Add-ons**: Device Protection, Tech Support, Online Security
* **Target Variable**: `Churn` (Yes / No)

---

## 🛠️ Technologies Used

* **Core Programming**: Python 3.10+
* **Machine Learning & Modeling**: Scikit-learn, XGBoost, Joblib
* **Data Processing & Analytics**: Pandas, NumPy
* **Interactive Frontend Dashboard**: Streamlit, Custom Dark CSS
* **Data Visualization**: Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
* **Database & SQL Engine**: PostgreSQL, SQLite, DuckDB SQL Scripts

---

## 🔄 ML Pipeline Architecture

```text
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  Raw Data Extraction │ ──>│  Data Cleaning &     │ ──>│  Feature Scaling &   │
│  (data/raw/csv)      │    │  Imputation          │    │  Label Encoding      │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
                                                                   │
┌──────────────────────┐    ┌──────────────────────┐               ▼
│  Streamlit App &     │ <──│  Hyperparameter      │ <──┌──────────────────────┐
│  Interactive UI      │    │  Tuning & ROC-AUC    │    │  Logistic Regression,│
└──────────────────────┘    └──────────────────────┘    │  Random Forest & XGB │
                                                        └──────────────────────┘
```

1. **Preprocessing & Imputation**: Fill missing Total Charges using `Tenure × Monthly Charges` logic; remove duplicates.
2. **Feature Engineering**: One-hot encode multi-categorical variables (`Payment Method`) and scale continuous numerical metrics using `StandardScaler`.
3. **Model Selection**: Train Logistic Regression, Random Forest, Gradient Boosting, and XGBoost classifiers.
4. **Segmentation**: Run K-Means clustering (3 clusters: Loyal Users, New Users, At-Risk Users).

---

## 🖼️ Dashboard Screenshots

### 1. Executive Dashboard Overview
![Executive Dashboard](dashboard_images/01_executive_dashboard.png)

### 2. Churn Risk Analysis
![Churn Risk Analysis](dashboard_images/02_churn_risk_analysis.png)

### 3. Customer Segmentation (K-Means)
![Customer Segmentation](dashboard_images/03_customer_segmentation.png)

### 4. Revenue Impact & Financial Loss
![Revenue Impact](dashboard_images/04_revenue_impact.png)

### 5. Real-Time Churn Predictor
![Prediction Screen](dashboard_images/05_prediction_screen.png)

---

## ⚙️ Installation & Setup Guide

### 1. Clone the repository
```bash
git clone https://github.com/abakaushik-lgtm/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📈 Results & Key Business Findings

### 🤖 Machine Learning Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 89.0% | 87.0% | 85.0% | 86.0% | 0.91 |
| **Random Forest** | 93.0% | 91.0% | 90.0% | 90.0% | 0.96 |
| **XGBoost Classifier** | **95.0%** | **94.0%** | **93.0%** | **93.0%** | **0.98** |

### Evaluation Visualizations
| Confusion Matrix | ROC Curve | Feature Importance |
|---|---|---|
| ![Confusion Matrix](dashboard_images/confusion_matrix.png) | ![ROC Curve](dashboard_images/roc_curve.png) | ![Feature Importance](dashboard_images/feature_importance.png) |

### 💡 Core Insights:
1. **Contract Type Vulnerability**: Month-to-Month contracts have a **42.7% churn rate** vs 2.8% for 2-year contracts.
2. **Tenure Stability Threshold**: Accounts active >24 months experience less than 10% churn probability.
3. **Support Ticket Alert**: Submitting 3+ support tickets indicates an 84% likelihood of imminent churn.

---

## 🔮 Future Improvements

* [ ] Integrate SHAP (SHapley Additive exPlanations) for local model interpretability per customer account.
* [ ] Implement automated email outreach API via Twilio / SendGrid for High-Risk accounts.
* [ ] Add Survival Analysis (Kaplan-Meier Curves) to predict exact time-to-churn.
* [ ] Deploy Docker containerization for Kubernetes production orchestration.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
