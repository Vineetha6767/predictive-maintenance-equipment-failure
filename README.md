# Predictive Maintenance & Equipment Failure Prediction System

## 📌 Project Overview

This project is a machine learning-based predictive maintenance system designed to predict the probability of equipment failure using machine operating conditions.

The system analyzes parameters such as temperature, rotational speed, torque, and tool wear to identify potential equipment failure risks before failure occurs.

A Streamlit web application is also developed to allow users to enter machine parameters and receive a failure probability and risk assessment.

---

## 🎯 Problem Statement

Unexpected equipment failures can lead to production downtime, maintenance costs, and operational losses.

Traditional maintenance approaches often depend on fixed schedules or maintenance after failure.

This project uses machine learning to analyze machine operating conditions and predict the probability of equipment failure, supporting a predictive maintenance approach.

---

## 🚀 Key Features

- Machine failure prediction using machine learning
- Handles imbalanced failure data
- Feature engineering for improved prediction
- Comparison of multiple machine learning models
- ROC-AUC and PR-AUC evaluation
- Confusion matrix and classification metrics
- Model feature importance analysis
- Failure risk classification
- Interactive Streamlit web application

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating conditions and failure information.

Important features include:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Target variable:

- `Machine failure`

Identifier columns such as `UDI` and `Product ID` were excluded from model training.

The failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`) were also excluded from the main predictive features to avoid target leakage.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🔧 Feature Engineering

Two additional features were created:

### Temperature Difference

```text
Temperature Difference =
Process Temperature - Air Temperature
```

### Mechanical Power

```text
Mechanical Power =
Torque × Rotational Speed
```

These features provide additional information about the machine's operating conditions.

---

## 🤖 Machine Learning Models

The following models were developed and compared:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used as a tree-based ensemble model for comparison.

### 3. XGBoost

Used as the final candidate model because of its strong performance on the predictive maintenance dataset.

Class imbalance was addressed using `scale_pos_weight` in XGBoost.

---

## ⚖️ Handling Class Imbalance

Machine failure is a relatively rare event in the dataset.

To address this imbalance, the training data was analyzed and XGBoost was configured using:

`scale_pos_weight`

This gives greater importance to the minority failure class and improves the model's ability to detect failures.

---

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For predictive maintenance, failure detection is particularly important, so recall and precision for the failure class were considered along with overall performance.

### Final XGBoost Model

The balanced XGBoost model achieved approximately:

- Accuracy: **97.5%**
- Failure Precision: **58.5%**
- Failure Recall: **91.2%**
- Failure F1-score: **71.3%**
- ROC-AUC: **0.978**
- PR-AUC: **0.888**

---

## 🔍 Model Explainability

XGBoost feature importance was used to understand which input features contributed most to the model's predictions.

The feature importance analysis helps identify the relative importance of machine operating parameters in the trained model.

Feature importance represents the model's internal usage of features and should not be interpreted as proof of causation.

---

## 🚦 Risk Assessment

The predicted failure probability is converted into three project-defined risk levels:

| Failure Probability | Risk Level |
|---|---|
| < 30% | Low Risk |
| 30% – 69.99% | Medium Risk |
| ≥ 70% | High Risk |

These thresholds are project-defined for the application and are not presented as universal industrial standards.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The application provides:

- Failure probability
- Risk level
- Model feature importance

---

## 📁 Project Structure

```text
predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   └── ai4i2020.csv
│
├── models/
│   └── xgb_balanced_pipeline.pkl
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── screenshots/
│
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md

---

## ▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/Vineetha6767/predictive-maintenance-equipment-failure.git
2. Navigate to the project directory
cd predictive-maintenance-equipment-failure
3. Create a virtual environment
python -m venv venv
4. Install dependencies
pip install -r requirements.txt
5. Run the Streamlit application
streamlit run app/app.py

The application will open in your browser.

--- 

## 📌 Future Enhancements
Deploy the application as a cloud-based web application
Add real-time machine sensor data
Add maintenance recommendations
Add historical failure monitoring
Improve risk threshold calibration using validation data
Add additional model monitoring capabilities

---

## 👩‍💻 Author

Vineetha

B.Tech – Computer Science (Data Science)

---

## 📄 License

This project is developed for educational and portfolio purposes.