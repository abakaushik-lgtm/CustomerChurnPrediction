-- ====================================================================
-- Query 06: Tenure Cohort & Monthly Retention Curve Trend
-- Description: Analyzes churn trajectory across tenure brackets (0-6m, 6-12m, 12-24m, 24m+)
-- Author: Product Analytics & Retention Strategy
-- ====================================================================

WITH TenureCohorts AS (
    SELECT 
        Customer_ID,
        CASE 
            WHEN Tenure <= 6 THEN '01. 0 - 6 Months'
            WHEN Tenure <= 12 THEN '02. 6 - 12 Months'
            WHEN Tenure <= 24 THEN '03. 12 - 24 Months'
            WHEN Tenure <= 48 THEN '04. 24 - 48 Months'
            ELSE '05. 48+ Months'
        END AS Tenure_Bucket,
        Monthly_Charges,
        Churn
    FROM customer_churn_data
)
SELECT 
    Tenure_Bucket,
    COUNT(Customer_ID) AS Cohort_Size,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Count,
    SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS Retained_Count,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN Monthly_Charges ELSE 0 END), 2) AS Monthly_Revenue_Loss
FROM TenureCohorts
GROUP BY Tenure_Bucket
ORDER BY Tenure_Bucket ASC;
