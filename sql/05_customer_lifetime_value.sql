-- ====================================================================
-- Query 05: Customer Lifetime Value (CLV) & Tier Segmentation
-- Description: Segments customer cohorts by lifetime revenue contribution and retention health
-- Author: Marketing & Growth Analytics Team
-- ====================================================================

WITH CLV_Calculation AS (
    SELECT 
        Customer_ID,
        Contract_Type,
        Tenure,
        Monthly_Charges,
        COALESCE(Total_Charges, Tenure * Monthly_Charges) AS Calculated_LTV,
        Churn,
        NTILE(4) OVER (ORDER BY COALESCE(Total_Charges, Tenure * Monthly_Charges) DESC) AS LTV_Quartile
    FROM customer_churn_data
)
SELECT 
    CASE 
        WHEN LTV_Quartile = 1 THEN 'Tier 1: High Value (Top 25%)'
        WHEN LTV_Quartile = 2 THEN 'Tier 2: Mid-High Value'
        WHEN LTV_Quartile = 3 THEN 'Tier 3: Mid-Low Value'
        ELSE 'Tier 4: Entry Level (Bottom 25%)'
    END AS LTV_Segment,
    COUNT(Customer_ID) AS Total_Customers,
    ROUND(AVG(Tenure), 1) AS Avg_Tenure_Months,
    ROUND(AVG(Monthly_Charges), 2) AS Avg_Monthly_Charges,
    ROUND(AVG(Calculated_LTV), 2) AS Avg_LTV,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS Churned_Count,
    ROUND((SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(Customer_ID)), 2) AS Churn_Rate_Pct
FROM CLV_Calculation
GROUP BY LTV_Quartile
ORDER BY LTV_Quartile ASC;
