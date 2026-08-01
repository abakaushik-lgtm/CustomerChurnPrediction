-- ====================================================================
-- Query 03: High-Risk Customer Identification & Priority Action Table
-- Description: Filters accounts with short tenure, high monthly charges, high ticket volume, and low engagement
-- Author: Customer Success & Analytics Team
-- ====================================================================

WITH RiskScoredCustomers AS (
    SELECT 
        Customer_ID,
        Gender,
        Contract_Type,
        Subscription_Type,
        Tenure,
        Monthly_Charges,
        Support_Tickets,
        Login_Frequency,
        Usage_Hours,
        Churn,
        -- Weighted composite risk index (0 to 100)
        ROUND(
            (CASE WHEN Contract_Type = 'Monthly' THEN 30 ELSE 5 END) +
            (CASE WHEN Support_Tickets >= 4 THEN 25 WHEN Support_Tickets >= 2 THEN 15 ELSE 5 END) +
            (CASE WHEN Login_Frequency < 5 THEN 25 WHEN Login_Frequency < 12 THEN 15 ELSE 5 END) +
            (CASE WHEN Tenure < 6 THEN 20 WHEN Tenure < 12 THEN 10 ELSE 0 END)
        , 2) AS Calculated_Risk_Score
    FROM customer_churn_data
)
SELECT 
    Customer_ID,
    Contract_Type,
    Subscription_Type,
    Tenure AS Tenure_Months,
    Monthly_Charges,
    Support_Tickets,
    Login_Frequency,
    Calculated_Risk_Score,
    CASE 
        WHEN Calculated_Risk_Score >= 70 THEN 'CRITICAL RISK'
        WHEN Calculated_Risk_Score >= 45 THEN 'HIGH RISK'
        WHEN Calculated_Risk_Score >= 25 THEN 'MEDIUM RISK'
        ELSE 'LOW RISK'
    END AS Priority_Tier,
    Churn AS Historical_Status
FROM RiskScoredCustomers
WHERE Calculated_Risk_Score >= 45 AND Churn = 'Yes' -- Or active accounts pending intervention
ORDER BY Calculated_Risk_Score DESC, Monthly_Charges DESC;
