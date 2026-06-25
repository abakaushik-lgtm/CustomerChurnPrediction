# Customer Churn Prediction & Analytics Platform

An intelligent, multi-tab customer churn prediction and behavior analytics platform. The application uses machine learning to classify churn risk probabilities, groups customers into behavioral segments using clustering, and provides personalized retention recommendations for customer success agents.

## 🚀 Key Features

* **Executive Dashboard**: Interactive KPI metrics cards, dynamic billing MRR churn simulator, and 3D segmentation charts.
* **Dataset Overview**: Automatic profiling of data duplicates, shapes, missing columns, and class imbalances, with raw data preview.
* **Churn Analytics (EDA)**: Beautiful, dark-themed Plotly charts showing demographics (contract and subscription type influence) and engagement correlations.
* **Model Studio**: Evaluates and compares **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **XGBoost**. Automatically selects the best-performing model based on F1-score, and renders ROC curves and Confusion Matrix plots.
* **Risk Assessment**: Detailed risk assessment database filtering accounts by risk level (Low, Medium, High).
* **Retention Recommendation Engine**: Dynamic business playbook lookup providing custom actions for customer success managers.
* **Real-time Predictor**: Sandbox tool allowing sales reps to enter customer attributes manually to run instant inferences.

## 📂 Project Structure

```text
├── generate_churn_data.py   # Compiles synthetic customer dataset with correlations
├── churn_engine.py          # Preprocessing, ML model training, and K-Means clustering pipelines
├── app.py                   # Streamlit interactive dark-themed dashboard frontend
├── customer_churn_data.csv  # Generated customer base dataset (7,043 rows, 20 features)
└── README.md                # Documentation guide
```

## 🛠️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/abakaushik-lgtm/CustomerChurnPrediction.git
cd CustomerChurnPrediction
```

### 2. Install dependencies
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly joblib
```

### 3. Generate the data (optional)
The dataset is generated automatically on the first run of the app, but you can also generate it manually:
```bash
python generate_churn_data.py
```

### 4. Run the Streamlit Dashboard
Launch the local web server:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to view the application.
