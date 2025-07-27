# Machine Learning Classification Projects: Loan Status and Diabetes Prediction

This repository contains two beginner-to-intermediate level classification machine learning projects:

1. Loan Status Prediction
2. Diabetes Prediction

Both projects follow standard machine learning workflows using Scikit-Learn and include data preprocessing, model training, and evaluation.

---

## Project 1: Loan Status Prediction

### Objective
Predict whether a loan application will be approved based on applicant information including credit history, income, education, and employment status.

### Dataset
A tabular dataset containing fields such as:
- Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area
- Loan_Status (target)

### Process Overview
- Load and clean data
- Replace and encode categorical features
- Impute missing values using mean and mode strategies
- Scale numerical features using StandardScaler
- Build and evaluate multiple classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost

### Evaluation
Models were evaluated using 5-fold cross-validation and accuracy score. The final evaluation includes a confusion matrix and classification report for the selected model.

### Key Libraries
- pandas, numpy, seaborn, matplotlib
- scikit-learn, xgboost

---

## Project 2: Diabetes Prediction

### Objective
Predict whether a patient is diabetic based on diagnostic attributes such as glucose level, BMI, insulin, and age.

### Dataset
PIMA Indians Diabetes Dataset with features including:
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age
- Outcome (target)

### Process Overview
- Basic data exploration and visualization
- Address zero values in important features (like Glucose, Insulin)
- Train a Support Vector Machine (SVM) classifier
- Evaluate performance on test data

### Evaluation
Model evaluated using accuracy score, confusion matrix, and classification report.

### Key Libraries
- pandas, numpy, seaborn, matplotlib
- scikit-learn

---

## Folder Structure

- `LOAN_STATUS_PREDICTION.ipynb` – Jupyter notebook for loan classification
- `diabetes_prediction.ipynb` – Jupyter notebook for diabetes classification
- `dataset.csv` – Dataset for loan project
- `diabetes.csv` – Dataset for diabetes project (if not provided, assume PIMA dataset from Kaggle or UCI)
- `README.md` – Project overview

---

## Notes

These projects are part of a training and learning phase in supervised machine learning. Future improvements could include:
- Adding SHAP/LIME for interpretability
- Hyperparameter tuning with GridSearchCV or Optuna
- Deployment with Streamlit or FastAPI
- Versioning and tracking with MLflow or DVC
