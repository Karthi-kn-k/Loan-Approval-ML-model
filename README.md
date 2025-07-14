# Loan-Approval-ML-model

# 🏦 Loan Approval Prediction App

This project is a Machine Learning web application that predicts whether a customer is likely to be approved for a loan, based on their demographic and financial information. It uses a Decision Tree Classifier and is built with **Streamlit** for the frontend interface.

---

## 📂 Dataset Source

This project uses the dataset from Kaggle:  
🔗 Loan Approval Prediction Dataset -->  ("https://www.kaggle.com/datasets/armanjitsingh/loan-approval-prediction-data")

The dataset has been preprocessed and cleaned before training.

---

## 📌 Features Used for Prediction

The model uses the following features to predict loan approval:

- `Gender` – Male / Female  
- `Married` – Yes / No  
- `Dependents` – Number of dependents (0, 1, 2, 3+)  
- `Education` – Graduate / Not Graduate  
- `Self_Employed` – Yes / No  
- `ApplicantIncome` – Applicant’s income  
- `CoapplicantIncome` – Co-applicant’s income  
- `LoanAmount` – Loan amount (in thousands)  
- `Loan_Amount_Term` – Term of the loan (in days)  
- `Credit_History` – 1: has credit history, 0: no credit history  
- `Property_Area` – Urban / Semiurban / Rural

The target variable is:

- `Loan_Status` – Whether the loan was approved (Y/N → 1/0)

---

## 📌 Overview Of The Model

- ✅ Predict loan approval with a Decision Tree model
- 📂 View preprocessed training dataset
- 📊 Display model evaluation metrics (Accuracy, Precision, Recall, F1 Score)
- 🌳 Visualize the top 3 levels of the trained decision tree
- 📥 Easy-to-use sidebar for user input
- 🎯 Clean and responsive Streamlit UI

---

## 🛠️ Tech Used

-> Python 3.x
-> Streamlit
-> Scikit-learn
-> Pandas
-> Matplotlib

---

## 📁 Project Structure

Loan-Approval-ML-model/
├── app.py # Streamlit app
├── data/
│ └── ProcessedLoan.csv # Cleaned Dataset
| |___RawLoan.csv #Default Dataset
├── outputs/ # Output images or files
├── scripts/ # Modular ML code
│ ├── preprocess.py
│ ├── train.py
│ ├── prediction.py
│ └── visualization.py
├── requirements.txt # Required packages
└── README.md # You're reading this! right now 😁