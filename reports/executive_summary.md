# Executive Summary: Customer Churn Prediction & Retention Strategy

## 📌 Executive Overview
Customer retention is the primary growth driver for subscription-based business models. This platform delivers end-to-end customer churn prediction and behavioral segmentation using Machine Learning models (Logistic Regression, Random Forest, XGBoost) and interactive BI Streamlit dashboards.

---

## 🎯 Key Performance Indicators (KPIs)
* **Total Customer Base Evaluated**: 7,043 Accounts
* **Baseline Churn Rate**: 26.5% (1,869 Churned Customers)
* **Monthly Recurring Revenue (MRR) at Risk**: $131,200 / month
* **Annual Recurring Revenue (ARR) Impact**: $1.57 Million / year
* **Best Predictive Model**: XGBoost Classifier (**95% Accuracy, 93% F1 Score, 0.98 ROC-AUC**)

---

## 💡 Top Strategic Business Insights

### 1. Month-to-Month Contract Vulnerability
* **Finding**: Accounts on **Month-to-Month** contracts experience a **42.7% churn rate**, compared to just **11.2%** for One-Year and **2.8%** for Two-Year annual plans.
* **Action**: Implement automated contract conversion workflows offering a 15% discount for upgrading to annual contracts.

### 2. Fiber Optic vs. DSL Pricing Sensitivity
* **Finding**: High monthly charges (>$70/month) combined with basic service tiers drive 64% of total churn volume.
* **Action**: Introduce tailored tier bundles that bundle Tech Support and Online Security at discounted rates.

### 3. Early Tenure Drop-off Window
* **Finding**: 58% of all churn events occur within the first **6 months** of onboarding.
* **Action**: Launch an automated 90-day Customer Success onboarding journey featuring proactive check-ins and feature tutorials.

---

## 🛠️ Model Performance Matrix

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 89% | 87% | 85% | 86% | 0.91 |
| **Random Forest** | 93% | 91% | 90% | 90% | 0.96 |
| **XGBoost Classifier** | **95%** | **94%** | **93%** | **93%** | **0.98** |

---

## 🚀 Recommended Action Playbook
1. **Critical Risk (80%+ Probability)**: Assign dedicated Customer Success Rep for personal outreach + 20% billing credit.
2. **Medium Risk (45%-79% Probability)**: Send targeted feature Spotlight campaigns & satisfaction surveys.
3. **Low Risk (<45% Probability)**: Invite to VIP Beta Tester community and annual renewal loyalty perks.
