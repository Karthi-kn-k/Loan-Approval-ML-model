import pandas as pd

def predict_new_customer(clf, input_dict, reference_columns):
    new_df = pd.DataFrame([input_dict])
    new_df = pd.get_dummies(new_df)
    new_df = new_df.reindex(columns=reference_columns, fill_value=0)
    prediction = clf.predict(new_df)[0]
    return prediction
