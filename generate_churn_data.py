import numpy as np
import pandas as pd
import os

def generate_data(file_path):
    np.random.seed(42)
    n_samples = 7043

    # Generate primary features
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_samples + 1)]
    genders = np.random.choice(["Male", "Female"], size=n_samples)
    ages = np.random.randint(18, 80, size=n_samples)
    senior_citizens = np.where(ages >= 65, np.random.choice([0, 1], p=[0.3, 0.7], size=n_samples), 0)
    partners = np.random.choice(["Yes", "No"], size=n_samples)
    dependents = np.random.choice(["Yes", "No"], p=[0.3, 0.7], size=n_samples)
    
    # Tenure: months with company
    tenures = np.random.geometric(p=0.03, size=n_samples)
    tenures = np.clip(tenures, 1, 72)  # Limit tenure to 1 to 72 months

    subscription_types = np.random.choice(["Basic", "Premium"], p=[0.6, 0.4], size=n_samples)
    contract_types = np.random.choice(["Monthly", "Annual"], p=[0.55, 0.45], size=n_samples)
    paperless_billings = np.random.choice(["Yes", "No"], p=[0.6, 0.4], size=n_samples)
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.4, 0.2, 0.2, 0.2],
        size=n_samples
    )

    # Monthly charges: Premium is higher
    base_charges = np.where(subscription_types == "Premium", 85.0, 35.0)
    noise_charges = np.random.normal(loc=10.0, scale=15.0, size=n_samples)
    monthly_charges = np.clip(base_charges + noise_charges, 18.0, 125.0).round(2)

    # Total charges: tenure * monthly charges
    total_charges = (tenures * monthly_charges).round(2)
    
    # Introduce some missing values in Total Charges (e.g., new customers with 0/1 month tenure or random)
    mask_missing = np.random.choice([True, False], p=[0.01, 0.99], size=n_samples)
    total_charges[mask_missing] = np.nan

    # Support tickets: higher for basic, higher monthly charges, and shorter tenure
    support_lambda = np.clip(2.5 + (monthly_charges / 50.0) - (tenures / 12.0), 0.5, 8.0)
    support_tickets = np.random.poisson(lam=support_lambda)

    # Login frequency: logins per month (range 1 to 30)
    # Loyal/Premium customers log in more. Churned customers log in less.
    login_mean = np.clip(15.0 + (tenures / 5.0) - support_tickets, 2.0, 28.0)
    login_frequencies = np.random.poisson(lam=login_mean)
    login_frequencies = np.clip(login_frequencies, 1, 30)

    # Usage hours: monthly hours (e.g. 5 to 250)
    usage_hours = np.clip((login_frequencies * np.random.uniform(4, 8, size=n_samples)), 2, 250).round(1)

    # Additional standard binary variables to reach 20 columns
    device_protections = np.random.choice(["Yes", "No"], p=[0.4, 0.6], size=n_samples)
    tech_supports = np.random.choice(["Yes", "No"], p=[0.35, 0.65], size=n_samples)
    online_securities = np.random.choice(["Yes", "No"], p=[0.3, 0.7], size=n_samples)

    # Calculate churn probability using a log-odds logit formula to make it realistic
    # We want Churn Rate around 26.5%
    # Logit formula:
    logit = (
        -0.5 
        - 0.04 * tenures 
        + 0.015 * monthly_charges
        + 0.25 * support_tickets
        - 0.15 * login_frequencies
        + 1.8 * (contract_types == "Monthly")
        + 0.5 * (subscription_types == "Basic")
        - 0.3 * (tech_supports == "Yes")
        - 0.3 * (online_securities == "Yes")
        + 0.4 * (ages > 60)
        - 0.002 * usage_hours
    )
    
    # Apply specific rule: login frequency < 5 logins/month has 3x higher churn probability
    # We adjust logit for low login frequency
    logit = np.where(login_frequencies < 5, logit + 1.5, logit)
    
    prob = 1 / (1 + np.exp(-logit))
    
    # Assign target variable Churn (Yes/No)
    churn_indicator = np.random.binomial(1, prob)
    churn = np.where(churn_indicator == 1, "Yes", "No")

    # Assemble dataframe
    df = pd.DataFrame({
        "Customer ID": customer_ids,
        "Gender": genders,
        "Age": ages,
        "Senior Citizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "Tenure": tenures,
        "Subscription Type": subscription_types,
        "Contract Type": contract_types,
        "Paperless Billing": paperless_billings,
        "Payment Method": payment_methods,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "Support Tickets": support_tickets,
        "Login Frequency": login_frequencies,
        "Usage Hours": usage_hours,
        "Device Protection": device_protections,
        "Tech Support": tech_supports,
        "Online Security": online_securities,
        "Churn": churn
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Dataset generated at: {file_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")

if __name__ == "__main__":
    generate_data("c:/Users/garvi/Documents/Data Science Projects/Customer Churn Prediction/customer_churn_data.csv")
