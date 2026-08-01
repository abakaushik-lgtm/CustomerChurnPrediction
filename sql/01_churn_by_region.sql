-- ====================================================================
-- Query 01: Customer Churn Analysis by Geographic Region / Payment Method
-- Description: Analyzes churn distribution across payment channels and service zones
-- Author: Data Science & BI Team
-- ====================================================================

WITH RegionalChurnMetrics AS (
    SELECT 
        Payment_Method AS Billing_Region,
        COUNT(Customer_ID) AS Total_Customers,
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Customers,
        SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) AS Retained_Customers,
        ROUND(AVG(Monthly_Charges), 2) AS Avg_Monthly_Charges,
        ROUND(SUM(Monthly_Charges), 2) AS Total_Monthly_Revenue,
        ROUND(SUM(CASE WHEN Churn = 'Yes' THEN Monthly_Charges ELSE 0 END), 2) AS Lost_Monthly_Revenue
    FROM customer_churn_data
    GROUP BY Payment_Method
)
SELECT 
    Billing_Region,
    Total_Customers,
    Churned_Customers,
    Retained_Customers,
    ROUND((Churned_Customers * 100.0 / Total_Customers), 2) AS Churn_Rate_Pct,
    Avg_Monthly_Charges,
    Total_Monthly_Revenue,
    Lost_Monthly_Revenue,
    ROUND((Lost_Monthly_Revenue * 100.0 / Total_Monthly_Revenue), 2) AS Revenue_Loss_Pct,
    RANK() OVER (ORDER BY (Churned_Customers * 100.0 / Total_Customers) DESC) AS Churn_Risk_Rank
FROM RegionalChurnMetrics
ORDER BY Churn_Rate_Pct DESC;
