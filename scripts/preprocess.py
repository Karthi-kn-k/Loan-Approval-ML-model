import pandas as pd

def load_and_preprocess_data(csv_path):
    data = pd.read_csv(csv_path)

    X = data.drop(columns='Loan_Status')
    y = data['Loan_Status']

    X = pd.get_dummies(X, drop_first=True)

    if y.dtype == 'object':
        y = y.map({'Y': 1, 'N': 0})

    return X, y
