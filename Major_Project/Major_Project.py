
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

url='https://drive.google.com/file/d/1TXr7b8FuZ4qb1fagnzXVwUYT9odrXk53/view?usp=drive_link'
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
dataset = pd.read_csv(dwn_url)

dataset.replace({'loan_status':{'Rejected':0,'Approved':1}},inplace=True)
dataset.replace({'loan_status':{'Rejected':0,'Approved':1},'self_employed':{'Yes':1,'No':0},'education':{'Graduate':1,'Not Graduate':0}},inplace=True)

X = dataset.drop(columns=['loan_id','loan_status'],axis=1)
Y = dataset['loan_status']

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=42)

best_model2 = RandomForestClassifier(
                                    max_depth=20, 
                                    min_samples_leaf=1, 
                                    min_samples_split=2, 
                                    n_estimators=100
                                    )
best_model2.fit(X_train,Y_train)
Y_pred_best_model2 = best_model2.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred_best_model2)
#print("Accuracy score of Best_Model 2 Random Forest: ", accuracy*100)
score = cross_val_score(best_model2,X,Y,cv=4)
#print("Cross Validation score of Best_Model 2 Random Forest: ",np.mean(score)*100)


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



st.write(f"Model accuracy (mean across 4 folds): {np.mean(score)*100:.2f}%")

if st.button("Enter"):
    prediction = best_model2.predict(df)
    if prediction == 1:
        st.success("Loan application is likely to be approved.")
    else:
        st.warning("Sorry, Loan application is likely to be rejected.")