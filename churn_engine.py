import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Try to import XGBoost, if not installed, we can fall back
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def clean_data(df):
    """
    Cleans raw customer churn data.
    Fills missing values and removes duplicate records.
    """
    cleaned_df = df.copy()
    
    # 1. Duplicate removal
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # 2. Missing value treatment
    # Total Charges has some missing values. We fill them with Tenure * Monthly Charges
    missing_total_charges = cleaned_df["Total Charges"].isna().sum()
    if missing_total_charges > 0:
        cleaned_df["Total Charges"] = cleaned_df.apply(
            lambda row: round(row["Tenure"] * row["Monthly Charges"], 2) 
            if pd.isna(row["Total Charges"]) else row["Total Charges"],
            axis=1
        )
        
    print(f"Data cleaning report: Duplicates removed = {duplicates_removed}, Missing Total Charges filled = {missing_total_charges}")
    return cleaned_df


def prepare_data(df, is_training=True, encoders=None, scaler=None):
    """
    Preprocesses raw columns: encodes categoricals, scales numericals, maps Target.
    """
    df_proc = df.copy()
    
    # List of features
    binary_cols = ["Gender", "Partner", "Dependents", "Subscription Type", "Contract Type", 
                   "Paperless Billing", "Device Protection", "Tech Support", "Online Security"]
    categorical_cols = ["Payment Method"]
    numeric_cols = ["Age", "Tenure", "Monthly Charges", "Total Charges", "Support Tickets", 
                    "Login Frequency", "Usage Hours", "Senior Citizen"]
    
    target_col = "Churn"
    
    # Map target if present
    y = None
    if target_col in df_proc.columns:
        y = df_proc[target_col].map({"Yes": 1, "No": 0}).values
        
    # Standardize binary columns to 0/1
    if is_training:
        encoders = {}
        for col in binary_cols:
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            encoders[col] = le
    else:
        for col in binary_cols:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels by mapping them to the first class
                df_proc[col] = df_proc[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
                df_proc[col] = le.transform(df_proc[col])
            else:
                # Default fallback label encoding
                df_proc[col] = df_proc[col].astype(str).map({"Yes": 1, "No": 0, "Male": 1, "Female": 0, "Basic": 0, "Premium": 1, "Monthly": 0, "Annual": 1}).fillna(0).astype(int)

    # One-hot encode multi-categorical variables
    df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=False)
    
    # Adjust for columns mismatch during prediction
    expected_ohe_cols = [
        "Payment Method_Electronic check", 
        "Payment Method_Mailed check", 
        "Payment Method_Bank transfer", 
        "Payment Method_Credit card"
    ]
    
    for col in expected_ohe_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    # Assemble feature columns list
    feature_cols = binary_cols + expected_ohe_cols + numeric_cols
    
    # Ensure all features exist in dataframe
    for col in feature_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0
            
    X = df_proc[feature_cols].copy()
    
    # Scale numeric columns
    if is_training:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        
    return X, y, scaler, encoders, feature_cols


def train_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Trains multiple classification models, evaluates them, and recommends the best one.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False, 
            eval_metric="logloss", 
            random_state=42
        )
        
    trained_models = {}
    metrics_summary = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC Curve & AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics_summary[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUC": roc_auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "confusion_matrix": cm.tolist()
        }
        
    # Recommend the best model based on F1 Score (standard for churn prediction)
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]["F1 Score"])
    
    return trained_models, metrics_summary, best_model_name


def run_kmeans_clustering(df):
    """
    Groups customers into 3 clusters using K-Means and profiles them dynamically.
    """
    df_clust = df.copy()
    
    # Numerical features to cluster on
    cluster_features = ["Tenure", "Monthly Charges", "Support Tickets", "Login Frequency", "Usage Hours"]
    
    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df_clust[cluster_features])
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_feats)
    df_clust["ClusterID"] = clusters
    
    # Map target Churn to numeric for profiling
    df_clust["ChurnNum"] = df_clust["Churn"].map({"Yes": 1, "No": 0})
    
    # Calculate profiles to map labels dynamically:
    # Segment definitions:
    # - At-Risk: Highest Churn Rate
    # - Loyal: Lowest Churn Rate & Highest Tenure
    # - New/Moderate: Remaining one
    profile = df_clust.groupby("ClusterID").agg({
        "ChurnNum": "mean",
        "Tenure": "mean",
        "Login Frequency": "mean"
    })
    
    # Find cluster IDs
    churn_rates = profile["ChurnNum"].to_dict()
    sorted_clusters_by_churn = sorted(churn_rates, key=churn_rates.get) # low to high
    
    loyal_cid = sorted_clusters_by_churn[0]
    at_risk_cid = sorted_clusters_by_churn[2]
    new_cid = sorted_clusters_by_churn[1]
    
    cluster_mapping = {
        loyal_cid: "Loyal Customers",
        new_cid: "New Customers",
        at_risk_cid: "At-Risk Customers"
    }
    
    # Store dynamic segments name
    df_clust["Segment"] = df_clust["ClusterID"].map(cluster_mapping)
    
    return kmeans, scaler, cluster_mapping, df_clust["Segment"].tolist(), profile


def get_retention_recommendations(row, churn_prob, risk_level):
    """
    Generates actionable business insights/actions based on risk score and key customer attributes.
    """
    actions = []
    
    if risk_level == "High Risk":
        if row.get("Contract Type") == "Monthly":
            actions.append("Offer Annual Contract Discount (Save 20% by shifting to annual billing).")
        if row.get("Support Tickets", 0) >= 3:
            actions.append("Schedule a priority Customer Success call to resolve technical pain points.")
        if row.get("Login Frequency", 30) < 5 or row.get("Usage Hours", 100) < 40:
            actions.append("Send re-engagement email campaign with free Premium subscription trial features.")
        if row.get("Subscription Type") == "Basic":
            actions.append("Offer subscription upgrade to Premium with a 3-month promotional discount.")
        
        # If no specific rule triggered
        if not actions:
            actions.append("Propose a direct 15% billing credit for the next 3 months to prevent immediate churn.")
            
    elif risk_level == "Medium Risk":
        if row.get("Support Tickets", 0) >= 1:
            actions.append("Send customer satisfaction survey and route negative feedback to Support Leads.")
        if row.get("Login Frequency", 30) < 12:
            actions.append("Send a personalized feature spotlight email to showcase unused platform capabilities.")
        if row.get("Contract Type") == "Monthly":
            actions.append("Promote the benefits of Annual subscription plans with a small loyalty bonus.")
        
        if not actions:
            actions.append("Perform a customer relationship review (check-in call or standard loyalty email).")
            
    else:  # Low Risk
        actions.append("Send standard newsletter and thank-you email containing a satisfaction survey.")
        if row.get("Usage Hours", 0) > 150:
            actions.append("Invite customer to join the VIP Beta Tester program for upcoming features.")
            
    return actions


def get_feature_importances(model, feature_names):
    """
    Extracts feature importances depending on model type.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For Logistic Regression, use absolute value of coefficients
        importances = np.abs(model.coef_[0])
        # Normalize to sum to 1
        if importances.sum() > 0:
            importances = importances / importances.sum()
    else:
        # Default fallback: uniform values
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return feat_imp


def compute_shap_explanations(model, X_sample, feature_names):
    """
    Calculates SHAP values if SHAP library is available.
    """
    if not SHAP_AVAILABLE:
        return None
        
    try:
        # TreeExplainer is fast for RF/XGBoost
        if model.__class__.__name__ in ["RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For Random Forest in sklearn, shap_values is a list for both classes.
            # We want the values for class 1 (churn).
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
                
            return shap_values
        else:
            # LinearExplainer or KernelExplainer fallbacks (Kernel can be slow, so limit sample size)
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            return shap_values.values
    except Exception as e:
        print(f"SHAP explanation computation failed: {e}")
        return None
