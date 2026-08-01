-- ====================================================================
-- Query 02: Churn Analysis by Subscription & Contract Type
-- Description: Evaluates how Month-to-Month vs Annual contracts impact customer retention
-- Author: Data Science & BI Team
-- ====================================================================

SELECT 
    Contract_Type,
    Subscription_Type,
    COUNT(Customer_ID) AS Total_Accounts,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Accounts,
    ROUND(AVG(Tenure), 1) AS Avg_Tenure_Months,
    ROUND(AVG(Monthly_Charges), 2) AS Avg_Monthly_Bill,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN Monthly_Charges ELSE 0 END), 2) AS MRR_Loss
FROM customer_churn_data
GROUP BY Contract_Type, Subscription_Type
ORDER BY Churn_Rate_Pct DESC;
