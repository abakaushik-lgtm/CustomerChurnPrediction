-- ====================================================================
-- Query 04: Revenue Impact & Financial Loss Analysis
-- Description: Quantifies Monthly Recurring Revenue (MRR) and Annual Recurring Revenue (ARR) lost to churn
-- Author: Finance & Revenue Intelligence Team
-- ====================================================================

SELECT 
    Subscription_Type,
    Contract_Type,
    COUNT(Customer_ID) AS Churned_Customer_Count,
    ROUND(SUM(Monthly_Charges), 2) AS Lost_MRR,
    ROUND(SUM(Monthly_Charges * 12), 2) AS Lost_ARR,
    ROUND(AVG(Monthly_Charges), 2) AS Avg_Lost_Monthly_Bill,
    ROUND(AVG(Total_Charges), 2) AS Avg_Lifetime_Spend_Prior_To_Churn
FROM customer_churn_data
WHERE Churn = 'Yes'
GROUP BY Subscription_Type, Contract_Type
ORDER BY Lost_MRR DESC;
