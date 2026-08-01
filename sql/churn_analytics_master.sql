-- ====================================================================
-- MASTER CHURN ANALYTICS SQL SUITE
-- Repository: Customer Churn Prediction & Analytics Platform
-- Description: Production-ready SQL scripts covering Customer Churn by Region, 
--              Contract Type, High-Risk Accounts, Revenue Impact, CLV, and Trends.
-- Compatible engines: PostgreSQL, MySQL, SQLite, DuckDB, Snowflake, BigQuery.
-- ====================================================================

-- 1. CHURN BY REGION / PAYMENT METHOD
CREATE VIEW IF NOT EXISTS view_churn_by_region AS
SELECT 
    Payment_Method AS Billing_Region,
    COUNT(Customer_ID) AS Total_Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Customers,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN Monthly_Charges ELSE 0 END), 2) AS Lost_Monthly_Revenue
FROM customer_churn_data
GROUP BY Payment_Method;

-- 2. CHURN BY CONTRACT TYPE
CREATE VIEW IF NOT EXISTS view_churn_by_contract AS
SELECT 
    Contract_Type,
    Subscription_Type,
    COUNT(Customer_ID) AS Total_Accounts,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Accounts,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct,
    ROUND(AVG(Tenure), 1) AS Avg_Tenure_Months
FROM customer_churn_data
GROUP BY Contract_Type, Subscription_Type;

-- 3. HIGH-RISK CUSTOMER IDENTIFICATION
CREATE VIEW IF NOT EXISTS view_high_risk_customers AS
SELECT 
    Customer_ID,
    Contract_Type,
    Subscription_Type,
    Tenure,
    Monthly_Charges,
    Support_Tickets,
    Login_Frequency,
    Churn
FROM customer_churn_data
WHERE Contract_Type = 'Monthly' 
  AND (Support_Tickets >= 3 OR Login_Frequency < 8)
  AND Churn = 'Yes';

-- 4. REVENUE LOST TO CHURN
CREATE VIEW IF NOT EXISTS view_revenue_lost AS
SELECT 
    Subscription_Type,
    COUNT(Customer_ID) AS Churned_Count,
    ROUND(SUM(Monthly_Charges), 2) AS Lost_MRR,
    ROUND(SUM(Monthly_Charges * 12), 2) AS Lost_ARR
FROM customer_churn_data
WHERE Churn = 'Yes'
GROUP BY Subscription_Type;

-- 5. CUSTOMER LIFETIME VALUE (CLV)
CREATE VIEW IF NOT EXISTS view_clv_segments AS
SELECT 
    Customer_ID,
    Tenure,
    Monthly_Charges,
    COALESCE(Total_Charges, Tenure * Monthly_Charges) AS Total_LTV,
    Churn
FROM customer_churn_data;

-- 6. MONTHLY TENURE CHURN TREND
CREATE VIEW IF NOT EXISTS view_tenure_churn_trend AS
SELECT 
    CASE 
        WHEN Tenure <= 6 THEN '0-6 Months'
        WHEN Tenure <= 12 THEN '6-12 Months'
        WHEN Tenure <= 24 THEN '12-24 Months'
        ELSE '24+ Months'
    END AS Tenure_Group,
    COUNT(Customer_ID) AS Total_Count,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Count,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct
FROM customer_churn_data
GROUP BY Tenure_Group;
