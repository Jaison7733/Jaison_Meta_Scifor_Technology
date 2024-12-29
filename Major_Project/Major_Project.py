
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'Model', 'rfclassifier.pkl') 
with open(model_path, 'rb') as f:
    rf_classifier = pickle.load(f)

st.title("Loan Prediction App")

Dependents = st.selectbox("Dependents", ["0", "1", "2", "3", "4", "5"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self_Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Yearly Applicant Income in Rs")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Amount Term in years", min_value=1, max_value=30)
Cibil_Score = st.number_input("Cibil Score",min_value=200, max_value=1000)
Residential_Assets_Value=st.number_input("Residential Assets Value")
Commercial_Assets_Value=st.number_input("Commercial Assets Value")
Luxury_Assets_Value=st.number_input("Luxury Assets Value")
Bank_Assets_Value=st.number_input("Bank Assets Value")

user_input = {

        'no_of_dependents': Dependents,
        'education': Education,
        'self_employed': Self_Employed,
        'income_annum': ApplicantIncome,
        'loan_amount': LoanAmount,
        'loan_term':Loan_Amount_Term,
        'cibil_score': Cibil_Score,
        'residential_assets_value': Residential_Assets_Value,
        'commercial_assets_value': Commercial_Assets_Value,
        'luxury_assets_value': Luxury_Assets_Value,
        'bank_asset_value': Bank_Assets_Value

        }

df = pd.DataFrame([user_input])

df.replace({'self_employed':{'Yes':1,'No':0},'education':{'Graduate':1,'Not Graduate':0}},inplace=True)

df = df.fillna(np.NAN)

#rf_classifier = pickle.load(open('Model/rfclassifier.pkl', 'rb'))



if st.button("Enter"):
    prediction = rf_classifier.predict(df)
    if prediction == 1:
        st.success("Loan application is likely to be approved.")
    else:
        st.warning("Sorry, Loan application is likely to be rejected.")