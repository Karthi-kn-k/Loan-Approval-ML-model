import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scripts.preprocess import load_and_preprocess_data
from scripts.train import train_decision_tree
from scripts.prediction import predict_new_customer
from scripts.visualization import plot_top3_tree

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

st.title("🏦 Loan Approval Predictor")
st.write("This app uses a decision tree model to predict whether a loan will be approved based on customer data.")

# 1. Load and preprocess
X, y = load_and_preprocess_data("data/ProcessedLoan.csv")
clf, x_train, x_test, y_train, y_test = train_decision_tree(X, y)

# 2. Show dataset before graph
st.subheader("📂 Processed Training Data")
st.dataframe(X.assign(Loan_Status=y))

# 3. Model Performance Metrics
st.subheader("📊 Model Performance Metrics")
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Accuracy", f"{acc*100:.2f}%")
col2.metric("🎯 Precision", f"{prec:.2f}")
col3.metric("🔁 Recall", f"{rec:.2f}")
col4.metric("📌 F1 Score", f"{f1:.2f}")

# 4. Sidebar input
st.sidebar.header("🔍 Enter Customer Details")
customer_input = {
    'Gender': st.sidebar.selectbox("Gender", ["Male", "Female"]),
    'Married': st.sidebar.selectbox("Married", ["Yes", "No"]),
    'Dependents': st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"]),
    'Education': st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"]),
    'Self_Employed': st.sidebar.selectbox("Self Employed", ["No", "Yes"]),
    'ApplicantIncome': st.sidebar.number_input("Applicant Income", min_value=0),
    'CoapplicantIncome': st.sidebar.number_input("Coapplicant Income", min_value=0.0),
    'LoanAmount': st.sidebar.number_input("Loan Amount", min_value=0.0),
    'Loan_Amount_Term': st.sidebar.number_input("Loan Term", min_value=0.0),
    'Credit_History': st.sidebar.selectbox("Credit History", [1.0, 0.0]),
    'Property_Area': st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
}

# 5. Predict loan approval
if st.sidebar.button("Predict Loan Approval"):
    prediction, input_encoded = predict_new_customer(clf, customer_input, X.columns)
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
    st.subheader("📌 Prediction Result")
    st.success(result)

    st.subheader("🧾 Input Features Sent to Model")
    st.write(input_encoded)

# 6. Visualize decision tree
st.subheader("🌳 Top 3 Levels of the Decision Tree")
fig = plot_top3_tree(clf, X.columns)
st.pyplot(fig)
