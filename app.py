import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as ob
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
import joblib

# Import custom engine functions
from churn_engine import (
    clean_data, prepare_data, train_models, run_kmeans_clustering,
    get_retention_recommendations, get_feature_importances,
    XGBOOST_AVAILABLE, SHAP_AVAILABLE
)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for stunning UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply font family globally without breaking Streamlit icons */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Customize main app background */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(to right, #38BDF8, #818CF8, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Metric Card styling */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    .kpi-card {
        flex: 1;
        min-width: 220px;
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
    }
    .kpi-card:hover {
        transform: translateY(-6px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.75rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .kpi-trend {
        font-size: 0.8rem;
        margin-top: 0.75rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    /* Gradient values */
    .val-blue { background: linear-gradient(135deg, #38BDF8, #0284C7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-indigo { background: linear-gradient(135deg, #818CF8, #4F46E5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-emerald { background: linear-gradient(135deg, #34D399, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-rose { background: linear-gradient(135deg, #F87171, #DC2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .val-amber { background: linear-gradient(135deg, #FBBF24, #D97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    /* Risk pill styling */
    .risk-badge {
        font-weight: 600;
        font-size: 0.8rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        display: inline-block;
        text-align: center;
    }
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .risk-badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Card panel styling */
    .card-panel {
        background: rgba(17, 24, 39, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #080C14;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
        color: #94A3B8;
        border-bottom-width: 2px;
        transition: all 0.2s ease;
    }
    button[data-baseweb="tab"]:hover {
        color: #38BDF8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #818CF8 !important;
        border-color: #818CF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Define file paths
DEFAULT_DATA_PATH = "c:/Users/garvi/Documents/Data Science Projects/Customer Churn Prediction/customer_churn_data.csv"

# Global helper to generate data if not present
if not os.path.exists(DEFAULT_DATA_PATH):
    from generate_churn_data import generate_data
    generate_data(DEFAULT_DATA_PATH)

# Initialize Session States
if "raw_df" not in st.session_state:
    st.session_state.raw_df = pd.read_csv(DEFAULT_DATA_PATH)
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = clean_data(st.session_state.raw_df)
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "metrics_summary" not in st.session_state:
    st.session_state.metrics_summary = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "encoders" not in st.session_state:
    st.session_state.encoders = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None
if "cluster_scaler" not in st.session_state:
    st.session_state.cluster_scaler = None
if "cluster_mapping" not in st.session_state:
    st.session_state.cluster_mapping = None

# Custom Model Training function cached in session state
def train_and_cache_models():
    df_clean = st.session_state.cleaned_df.copy()
    
    # Preprocess
    X, y, scaler, encoders, feature_cols = prepare_data(df_clean, is_training=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    trained, metrics, best_name = train_models(X_train, y_train, X_test, y_test, feature_cols)
    
    # Run K-Means Customer Segmentation
    kmeans, clust_scaler, clust_map, segments, profile = run_kmeans_clustering(df_clean)
    
    # Store in session state
    st.session_state.trained_models = trained
    st.session_state.metrics_summary = metrics
    st.session_state.best_model_name = best_name
    st.session_state.scaler = scaler
    st.session_state.encoders = encoders
    st.session_state.feature_names = feature_cols
    st.session_state.kmeans_model = kmeans
    st.session_state.cluster_scaler = clust_scaler
    st.session_state.cluster_mapping = clust_map
    
    # Add segments to cleaned dataframe
    st.session_state.cleaned_df["Segment"] = segments
    
    # Add predictions & risk scores to cleaned dataframe
    best_model = trained[best_name]
    # Predict probabilities for the entire dataset
    X_all, _, _, _, _ = prepare_data(df_clean, is_training=False, encoders=encoders, scaler=scaler)
    probs = best_model.predict_proba(X_all)[:, 1]
    
    st.session_state.cleaned_df["Churn Probability"] = probs
    
    # Map Risk levels
    def get_risk_tier(p):
        if p <= 0.30: return "Low Risk"
        elif p <= 0.70: return "Medium Risk"
        else: return "High Risk"
        
    st.session_state.cleaned_df["Risk Level"] = st.session_state.cleaned_df["Churn Probability"].apply(get_risk_tier)

# Train on first load automatically
if st.session_state.trained_models is None:
    train_and_cache_models()

# App Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">🔮 Customer Churn Analytics Platform</h1>
    <p class="app-subtitle">Intelligent behavior analysis, predictive risk modeling, and retention strategy automation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Layout
with st.sidebar:
    st.markdown("### 📊 Platform Control Center")
    
    # Upload Dataset feature
    uploaded_file = st.file_uploader("Upload customer CSV dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            # Verify columns match basic shape
            required_cols = ["Gender", "Age", "Tenure", "Subscription Type", "Contract Type", "Monthly Charges", "Total Charges", "Churn"]
            if all(col in uploaded_df.columns for col in required_cols):
                st.session_state.raw_df = uploaded_df
                st.session_state.cleaned_df = clean_data(uploaded_df)
                train_and_cache_models()
                st.success("New dataset loaded and models successfully retrained!")
            else:
                st.error("Uploaded CSV is missing critical churn columns. Please check your data format.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 Predictive Model Status")
    st.info(f"Active Model: **{st.session_state.best_model_name}**")
    
    # Quick Action - Re-train models
    if st.button("🔄 Retrain ML Models"):
        with st.spinner("Re-training ML pipelines..."):
            train_and_cache_models()
        st.success("Models retrained successfully!")
        
    st.markdown("---")
    
    # Quick risk summary
    st.markdown("### ⚠️ Customer Risk Distribution")
    risk_counts = st.session_state.cleaned_df["Risk Level"].value_counts()
    
    low_count = risk_counts.get("Low Risk", 0)
    med_count = risk_counts.get("Medium Risk", 0)
    high_count = risk_counts.get("High Risk", 0)
    total_cust = len(st.session_state.cleaned_df)
    
    st.markdown(f"""
    <div style='display: flex; flex-direction: column; gap: 0.5rem;'>
        <div style='display: flex; justify-content: space-between;'>
            <span>🟢 Low Risk (0-30%):</span>
            <span class='risk-badge risk-badge-low'>{low_count} ({low_count/total_cust*100:.1f}%)</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span>🟡 Medium Risk (31-70%):</span>
            <span class='risk-badge risk-badge-medium'>{med_count} ({med_count/total_cust*100:.1f}%)</span>
        </div>
        <div style='display: flex; justify-content: space-between;'>
            <span>🔴 High Risk (71-100%):</span>
            <span class='risk-badge risk-badge-high'>{high_count} ({high_count/total_cust*100:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.8rem;'>Powered by Churn Prediction Engine v1.0.0</p>", unsafe_allow_html=True)

# Main Dashboard Tabs
tabs = st.tabs([
    "📈 Executive Dashboard",
    "📂 Dataset Overview",
    "📊 Churn Analytics",
    "🤖 Churn Prediction",
    "⚠️ Risk Assessment",
    "🎯 Retention Strategies",
    "🔮 Real-Time Predictor"
])

# Variables for KPI calculations
df_view = st.session_state.cleaned_df
churn_rate_val = (df_view["Churn"] == "Yes").mean() * 100
total_customers_val = len(df_view)
active_customers_val = (df_view["Churn"] == "No").sum()
monthly_revenue_lost_val = df_view[df_view["Churn"] == "Yes"]["Monthly Charges"].sum()
retention_rate_val = 100 - churn_rate_val

# ----------------- TAB 0: EXECUTIVE DASHBOARD -----------------
with tabs[0]:
    # Rich KPI cards using Custom HTML grid
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value val-blue">{total_customers_val:,}</div>
            <div class="kpi-trend" style="color: #94A3B8;">👥 Active subscribers database</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Active Customers</div>
            <div class="kpi-value val-indigo">{active_customers_val:,}</div>
            <div class="kpi-trend" style="color: #10B981;">🟢 {active_customers_val/total_customers_val*100:.1f}% of total base</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value val-rose">{churn_rate_val:.2f}%</div>
            <div class="kpi-trend" style="color: #EF4444;">🔴 Target < 10%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Revenue Lost (Monthly)</div>
            <div class="kpi-value val-amber">${monthly_revenue_lost_val:,.2f}</div>
            <div class="kpi-trend" style="color: #FBBF24;">📉 Impact of customer attrition</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Retention Rate</div>
            <div class="kpi-value val-emerald">{retention_rate_val:.2f}%</div>
            <div class="kpi-trend" style="color: #10B981;">📈 Monthly retention KPI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.subheader("💡 Business Revenue Impact Simulator")
        st.write("Simulate how improving retention among High-Risk customers impacts monthly recurring revenue (MRR).")
        
        # Slider for simulation
        target_retention = st.slider(
            "Prevent Churn Rate among High Risk Customers (%)", 
            min_value=0, 
            max_value=100, 
            value=25, 
            step=5
        )
        
        # Calculate simulation values
        high_risk_df = df_view[df_view["Risk Level"] == "High Risk"]
        high_risk_mrr = high_risk_df["Monthly Charges"].sum()
        revenue_saved = (target_retention / 100.0) * high_risk_mrr
        new_revenue_lost = monthly_revenue_lost_val - revenue_saved
        new_churn_rate = ((df_view["Churn"] == "Yes").sum() - (len(high_risk_df) * (target_retention/100.0))) / total_customers_val * 100
        new_churn_rate = max(0.0, new_churn_rate)
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.metric(
                label="Estimated Monthly Revenue Saved",
                value=f"${revenue_saved:,.2f}",
                delta=f"{target_retention}% Saved",
                delta_color="normal"
            )
        with sim_col2:
            st.metric(
                label="Simulated Monthly Revenue Lost",
                value=f"${new_revenue_lost:,.2f}",
                delta=f"-${revenue_saved:,.2f} reduction",
                delta_color="inverse"
            )
            
        st.info(f"Saving {target_retention}% of high-risk accounts reduces the overall churn rate from **{churn_rate_val:.2f}%** down to **{new_churn_rate:.2f}%**.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.subheader("👥 Customer Segments Profile (K-Means)")
        st.write("Distribution of customers based on engagement, platform usage, support tickets, and monthly bills.")
        
        # Visualize cluster sizes and profiles
        segment_counts = df_view["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Count"]
        
        fig_seg = px.pie(
            segment_counts, 
            values="Count", 
            names="Segment", 
            hole=0.4,
            color="Segment",
            color_discrete_map={
                "Loyal Customers": "#10B981",
                "New Customers": "#38BDF8",
                "At-Risk Customers": "#EF4444"
            },
            template="plotly_dark"
        )
        fig_seg.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
        st.plotly_chart(fig_seg, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3D Customer Segmentation Scatter Plot
    st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
    st.subheader("📍 Interactive 3D Segment Visualization")
    st.write("Rotate and zoom the 3D plot of clusters based on Tenure, Support Tickets, and Monthly Charges.")
    
    fig_3d = px.scatter_3d(
        df_view,
        x="Tenure",
        y="Monthly Charges",
        z="Support Tickets",
        color="Segment",
        opacity=0.6,
        color_discrete_map={
            "Loyal Customers": "#10B981",
            "New Customers": "#38BDF8",
            "At-Risk Customers": "#EF4444"
        },
        template="plotly_dark",
        height=500
    )
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Tenure (Months)",
            yaxis_title="Monthly Charges ($)",
            zaxis_title="Support Tickets"
        ),
        margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 1: DATASET OVERVIEW -----------------
with tabs[1]:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.subheader("📊 Automatic Profiling Metrics")
        
        # Profile details
        profiling_metrics = {
            "Total Records": len(st.session_state.raw_df),
            "Number of Columns": len(st.session_state.raw_df.columns),
            "Missing Values Detected": st.session_state.raw_df.isna().sum().sum(),
            "Duplicate Records": st.session_state.raw_df.duplicated().sum(),
            "Imbalance (Churn Rate)": f"{(st.session_state.raw_df['Churn'] == 'Yes').mean() * 100:.1f}% Churn vs { (st.session_state.raw_df['Churn'] == 'No').mean() * 100:.1f}% Retain"
        }
        
        for k, v in profiling_metrics.items():
            st.markdown(f"**{k}**: `{v}`")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.subheader("🧹 Automated Data Cleaning")
        st.write("""
        1. **Missing values**: `Total Charges` missing values are imputed dynamically using formula `Tenure * Monthly Charges`.
        2. **Duplicates**: Duplicate customer entries are automatically purged.
        3. **Normalization**: Numerical columns are scaled using `StandardScaler` for neural/classification model consistency.
        4. **Encoding**: Categorical binary classes (e.g. Contract, Gender) are encoded to binary vectors (0/1). Multi-class variables are one-hot encoded.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.subheader("🔍 Dataset Inspector")
        st.write("Browse raw data entries and attributes:")
        st.dataframe(st.session_state.raw_df.head(100), height=350)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: CHURN ANALYTICS (EDA) -----------------
with tabs[2]:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    eda_tab_sel = st.selectbox("Choose EDA Focus", [
        "Churn Distribution & Revenue", 
        "Customer Demographics Analysis", 
        "Engagement & Product Usage"
    ])
    
    if eda_tab_sel == "Churn Distribution & Revenue":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Churn Class Distribution")
            churn_counts = df_view["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Status", "Count"]
            
            fig = px.pie(
                churn_counts, 
                values="Count", 
                names="Status", 
                hole=0.4,
                color="Status",
                color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Monthly Revenue Distribution ($)")
            
            # Monthly Revenue vs Churn Boxplot
            fig_box = px.box(
                df_view,
                x="Churn",
                y="Monthly Charges",
                color="Churn",
                color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
                points="outliers",
                template="plotly_dark"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    elif eda_tab_sel == "Customer Demographics Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Churn by Contract Type")
            # Group by contract and churn
            contract_churn = df_view.groupby(["Contract Type", "Churn"]).size().reset_index(name="Count")
            
            fig_bar = px.bar(
                contract_churn, 
                x="Contract Type", 
                y="Count", 
                color="Churn",
                barmode="group",
                color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Churn by Subscription Type")
            sub_churn = df_view.groupby(["Subscription Type", "Churn"]).size().reset_index(name="Count")
            
            fig_sub = px.bar(
                sub_churn,
                x="Subscription Type",
                y="Count",
                color="Churn",
                barmode="group",
                color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_sub, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Age distribution
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Age Distribution vs Churn Status")
        fig_age = px.histogram(
            df_view,
            x="Age",
            color="Churn",
            barmode="overlay",
            marginal="box",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif eda_tab_sel == "Engagement & Product Usage":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Churn Rate by Support Tickets")
            # Support tickets vs Churn rate
            ticket_churn = df_view.groupby("Support Tickets")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).reset_index(name="Churn Rate (%)")
            
            fig_ticket = px.line(
                ticket_churn,
                x="Support Tickets",
                y="Churn Rate (%)",
                markers=True,
                line_shape="linear",
                template="plotly_dark"
            )
            fig_ticket.update_traces(line_color="#F59E0B")
            st.plotly_chart(fig_ticket, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### Login Frequency vs Churn")
            # High logins vs low logins
            fig_login = px.histogram(
                df_view,
                x="Login Frequency",
                color="Churn",
                barmode="group",
                color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
                template="plotly_dark"
            )
            st.plotly_chart(fig_login, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Usage Hours vs Monthly Charges Scatter")
        fig_scatter = px.scatter(
            df_view,
            x="Usage Hours",
            y="Monthly Charges",
            color="Churn",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            opacity=0.5,
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: CHURN PREDICTION (MODELS) -----------------
with tabs[3]:
    st.subheader("🤖 Churn Prediction Model Studio")
    st.write("Compare classification algorithms, analyze performance metrics, and visualize features driving customer churn.")
    
    metrics_summary = st.session_state.metrics_summary
    trained_models = st.session_state.trained_models
    best_model_name = st.session_state.best_model_name
    
    if metrics_summary is not None:
        # Comparison Table
        comparison_data = []
        for model_name, metrics in metrics_summary.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
                "Precision": f"{metrics['Precision']*100:.2f}%",
                "Recall": f"{metrics['Recall']*100:.2f}%",
                "F1 Score": f"{metrics['F1 Score']*100:.2f}%",
                "AUC-ROC": f"{metrics['AUC']*100:.2f}%",
                "Recommendation Status": "⭐ Recommended Best Model" if model_name == best_model_name else "Candidate Model"
            })
            
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write("#### ROC (Receiver Operating Characteristic) Curve Comparison")
            
            # Interactive ROC plot
            fig_roc = ob.Figure()
            for model_name, metrics in metrics_summary.items():
                fig_roc.add_trace(ob.Scatter(
                    x=metrics["fpr"],
                    y=metrics["tpr"],
                    mode="lines",
                    name=f"{model_name} (AUC = {metrics['AUC']:.3f})"
                ))
            fig_roc.add_trace(ob.Scatter(
                x=[0, 1], y=[0, 1], 
                mode="lines", 
                line=dict(dash="dash", color="grey"), 
                name="Random Guess"
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_dark"
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
            st.write(f"#### Confusion Matrix - {best_model_name}")
            
            cm = metrics_summary[best_model_name]["confusion_matrix"]
            
            # Heatmap confusion matrix
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Customers"),
                x=["Retain (0)", "Churn (1)"],
                y=["Retain (0)", "Churn (1)"],
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig_cm.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Feature Importance Section
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### ⚡ Drivers of Churn: Feature Importance")
        
        # Get importances for best model
        best_model = trained_models[best_model_name]
        feat_imp = get_feature_importances(best_model, st.session_state.feature_names)
        
        # Plotly Horizontal Bar
        fig_imp = px.bar(
            feat_imp.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Bluered",
            template="plotly_dark"
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        st.warning("Please verify models are trained.")

# ----------------- TAB 4: RISK ASSESSMENT -----------------
with tabs[4]:
    st.subheader("⚠️ Customer Risk Assessment Center")
    st.write("Detailed view of specific customer risk profiles. High risk scores demand proactive retention outreach.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Risk Filter Settings")
        
        risk_filter = st.radio(
            "Select Risk Tier to View:", 
            ["All Risk Tiers", "High Risk 🔴 (71-100%)", "Medium Risk 🟡 (31-70%)", "Low Risk 🟢 (0-30%)"]
        )
        
        min_prob, max_prob = st.slider(
            "Filter by Churn Probability Range:",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Risk Profiles Table")
        
        # Apply filters
        df_risk_filtered = df_view.copy()
        
        # Risk tier filter
        if "High Risk" in risk_filter:
            df_risk_filtered = df_risk_filtered[df_risk_filtered["Risk Level"] == "High Risk"]
        elif "Medium Risk" in risk_filter:
            df_risk_filtered = df_risk_filtered[df_risk_filtered["Risk Level"] == "Medium Risk"]
        elif "Low Risk" in risk_filter:
            df_risk_filtered = df_risk_filtered[df_risk_filtered["Risk Level"] == "Low Risk"]
            
        # Probability slider filter
        df_risk_filtered = df_risk_filtered[
            (df_risk_filtered["Churn Probability"] >= min_prob) & 
            (df_risk_filtered["Churn Probability"] <= max_prob)
        ]
        
        # Columns to display
        disp_cols = [
            "Customer ID", "Gender", "Age", "Tenure", 
            "Contract Type", "Monthly Charges", "Support Tickets", 
            "Login Frequency", "Segment", "Churn Probability", "Risk Level"
        ]
        
        # Format columns for beautiful display
        df_risk_disp = df_risk_filtered[disp_cols].copy()
        df_risk_disp["Churn Probability"] = df_risk_disp["Churn Probability"].apply(lambda p: f"{p*100:.1f}%")
        
        st.dataframe(df_risk_disp.sort_values(by="Churn Probability", ascending=False), use_container_width=True, height=400)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 5: RETENTION STRATEGIES -----------------
with tabs[5]:
    st.subheader("🎯 Automated Retention Recommendation Engine")
    st.write("Generate targeted business actions for customer success reps based on automated risk diagnostics.")
    
    # Filter to medium and high risk customers for action
    df_actions = df_view[df_view["Risk Level"].isin(["High Risk", "Medium Risk"])].copy()
    
    if len(df_actions) == 0:
        st.success("🎉 Outstanding! No medium or high risk customer accounts detected in the current filter.")
    else:
        st.info(f"Generated actionable strategies for **{len(df_actions)}** at-risk customer accounts.")
        
        # Generate recommendation list dynamically
        recs_list = []
        for idx, row in df_actions.iterrows():
            actions = get_retention_recommendations(row, row["Churn Probability"], row["Risk Level"])
            recs_list.append({
                "Customer ID": row["Customer ID"],
                "Risk Score": f"{row['Churn Probability']*100:.1f}%",
                "Risk Tier": row["Risk Level"],
                "Contract Type": row["Contract Type"],
                "Support Tickets": row["Support Tickets"],
                "Suggested Actions": " | ".join(actions)
            })
            
        recs_df = pd.DataFrame(recs_list)
        
        # Selectbox to inspect single customer actions in detail
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("### 🔍 Customer Retention Playbook Lookup")
        cust_sel = st.selectbox(
            "Select Customer ID to fetch direct retention playbook:", 
            recs_df["Customer ID"].unique()
        )
        
        cust_row = df_actions[df_actions["Customer ID"] == cust_sel].iloc[0]
        cust_rec = recs_df[recs_df["Customer ID"] == cust_sel].iloc[0]
        
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown(f"""
            * **Age / Gender**: {cust_row['Age']} yrs / {cust_row['Gender']}
            * **Tenure**: {cust_row['Tenure']} months
            * **Contract**: {cust_row['Contract Type']}
            * **Billing Rate**: ${cust_row['Monthly Charges']}/month
            * **Support Tickets**: {cust_row['Support Tickets']} complaints
            """)
        with col_det2:
            st.markdown(f"**Calculated Churn Risk**: `{cust_rec['Risk Score']}` ({cust_rec['Risk Tier']})")
            st.markdown("**Tailored Customer Success Plays:**")
            plays = cust_rec["Suggested Actions"].split(" | ")
            for play in plays:
                st.markdown(f"- `{play}`")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Export option
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### 📁 Export Playbook Database")
        csv_data = recs_df.to_csv(index=False)
        st.download_button(
            label="Download Retention Action Plan (CSV)",
            data=csv_data,
            file_name="customer_retention_recommendations.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 6: REAL-TIME PREDICTOR -----------------
with tabs[6]:
    st.subheader("🔮 Customer Churn Risk Estimator")
    st.write("Manually enter a customer's usage, contract, and billing details to estimate churn probability instantly.")
    
    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Demographics & Subscriptions")
        
        val_gender = st.selectbox("Gender", ["Female", "Male"])
        val_age = st.slider("Customer Age", 18, 90, 35)
        val_senior = st.selectbox("Senior Citizen (>= 65)", [0, 1])
        val_partner = st.selectbox("Partner Status", ["Yes", "No"])
        val_dependents = st.selectbox("Dependents Status", ["Yes", "No"])
        val_tenure = st.slider("Tenure (Months)", 1, 72, 12)
        val_sub = st.selectbox("Subscription Plan", ["Basic", "Premium"])
        val_contract = st.selectbox("Contract Type", ["Monthly", "Annual"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_in2:
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.write("#### Billing & Platform Engagement")
        
        val_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        val_pay_method = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        val_monthly = st.slider("Monthly charges ($)", 18, 125, 65)
        val_total = st.number_input("Total Charges ($)", min_value=18.0, value=float(val_tenure * val_monthly))
        val_tickets = st.slider("Support Tickets Opened", 0, 10, 2)
        val_logins = st.slider("Login Frequency (Per Month)", 1, 30, 12)
        val_usage = st.slider("Usage Hours (Per Month)", 5.0, 250.0, 85.0)
        
        # Additional fields
        val_device_prot = st.selectbox("Device Protection", ["Yes", "No"])
        val_tech_sup = st.selectbox("Tech Support Premium Package", ["Yes", "No"])
        val_sec = st.selectbox("Online Security Package", ["Yes", "No"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Prediction trigger
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Calculate Customer Churn Risk Probability", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if predict_btn:
        # Prepare inputs dictionary
        input_data = pd.DataFrame([{
            "Gender": val_gender,
            "Age": val_age,
            "Senior Citizen": val_senior,
            "Partner": val_partner,
            "Dependents": val_dependents,
            "Tenure": val_tenure,
            "Subscription Type": val_sub,
            "Contract Type": val_contract,
            "Paperless Billing": val_paperless,
            "Payment Method": val_pay_method,
            "Monthly Charges": val_monthly,
            "Total Charges": val_total,
            "Support Tickets": val_tickets,
            "Login Frequency": val_logins,
            "Usage Hours": val_usage,
            "Device Protection": val_device_prot,
            "Tech Support": val_tech_sup,
            "Online Security": val_sec
        }])
        
        # Preprocess using loaded scaling assets
        X_inp, _, _, _, _ = prepare_data(
            input_data, 
            is_training=False, 
            encoders=st.session_state.encoders, 
            scaler=st.session_state.scaler
        )
        
        # Inference using recommended best model
        best_model = st.session_state.trained_models[st.session_state.best_model_name]
        prob = best_model.predict_proba(X_inp)[0, 1]
        
        # Risk tier calculation
        risk_name = "Low Risk"
        risk_class = "risk-badge-low"
        risk_color = "🟢"
        if prob > 0.70:
            risk_name = "High Risk"
            risk_class = "risk-badge-high"
            risk_color = "🔴"
        elif prob > 0.30:
            risk_name = "Medium Risk"
            risk_class = "risk-badge-medium"
            risk_color = "🟡"
            
        # Display Result
        st.markdown("<div class='card-panel'>", unsafe_allow_html=True)
        st.markdown(f"### Churn Diagnosis Summary")
        
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.metric(
                label="Calculated Churn Probability",
                value=f"{prob*100:.1f}%",
                delta=f"Risk Level: {risk_color} {risk_name}"
            )
        with det_col2:
            st.write("#### Actionable Playbook Recommended:")
            plays = get_retention_recommendations(
                input_data.iloc[0].to_dict(), 
                prob, 
                risk_name
            )
            for play in plays:
                st.markdown(f"- `{play}`")
        st.markdown("</div>", unsafe_allow_html=True)
