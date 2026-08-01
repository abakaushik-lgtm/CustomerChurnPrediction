# Detailed Customer Churn Analytics & Behavioral Insights Report

## 📖 Introduction
This report details the data-driven findings from analyzing 7,043 customer accounts across demographics, service subscriptions, payment behaviors, and technical support interactions.

---

## 🔍 Detailed Analytical Findings

### 1. Contract & Billing Influence
* **Month-to-Month Contracts**: Account for **74% of all churned customers**. Short-term flexibility increases churn propensity by **3.8x**.
* **Payment Methods**: Electronic Check users experience **32.8% churn rate**, whereas Direct Bank Transfer and Credit Card auto-pay users churn at **<15%**.

### 2. Service Add-Ons & Risk Mitigation
* **Tech Support**: Customers *with* active Tech Support churn at only **15.1%**, versus **31.4%** for customers *without* Tech Support.
* **Online Security**: Adding Online Security reduces overall churn risk by **over 30%**.

### 3. Customer Engagement Metrics
* **Support Ticket Volume**: Customers logging 3 or more support tickets within 60 days have an **84% likelihood of churn**.
* **Login Frequency**: Accounts logging in fewer than 5 times per month churn at **3x the rate** of highly active users (>15 logins/mo).

---

## 📉 Revenue Loss Breakdown

```text
+------------------------+-------------------+-----------------+
| Subscription Tier      | Churned Accounts  | Lost MRR ($)    |
+------------------------+-------------------+-----------------+
| Basic Monthly          | 582               | $42,500         |
| Basic Annual           | 114               | $8,200          |
| Premium Monthly        | 948               | $68,400         |
| Premium Annual         | 225               | $12,100         |
+------------------------+-------------------+-----------------+
| TOTAL                  | 1,869             | $131,200        |
+------------------------+-------------------+-----------------+
```

---

## 🎯 Strategic Recommendations
1. **Targeted Annual Upgrades**: Incentivize Month-to-Month users on Premium plans to upgrade to Annual plans.
2. **Support Ticket Alert Trigger**: Automatically route accounts submitting 2+ support tickets within 30 days to Tier-2 Priority Support.
3. **Re-engagement Campaign**: Deploy automated emails offering product walk-through webinars for accounts with declining login frequencies (<5 logins/mo).
