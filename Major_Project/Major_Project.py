import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
loan_dataset = pd.read_csv(uploaded_file)
loan_dataset=loan_dataset.dropna()
loan_dataset.replace({"Loan_Status":{"N":0,"Y":1}},inplace=True)
loan_dataset.replace(to_replace="3+",value=4,inplace=True)
loan_dataset["Total_Applicant_Income"]=loan_dataset["ApplicantIncome"]+loan_dataset["CoapplicantIncome"]
#drop unnecessary columns
cols=["CoapplicantIncome","Loan_Amount_Term"]
loan_dataset=loan_dataset.drop(columns=cols,axis=1)
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=42)
# Create a Random Forest Classifier object
rf_classifier = RandomForestClassifier(n_estimators=40, random_state=42) 
# Train the model
rf_classifier.fit(X_train, Y_train)
# accuracy score on training data
X_train_prediction = rf_classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)
#print('Accuracy on training data : ', training_data_accuray)
# accuracy score on training data
X_test_prediction = rf_classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
#print('Accuracy on test data : ', test_data_accuray)




st.title("Loan Prediction App")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self_Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income in $")
CoapplicantIncome = st.number_input("Coapplicant Income in $")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Amount Term in months", min_value=12, max_value=360)
Credit_History = st.selectbox("Credit History (Good=1, Bad=0)", ["1", "0"])
Property_Area = st.selectbox("Property_Area", ["Rural", "Urban", "Semiurban"])

user_input = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area

        }

df = pd.DataFrame([user_input])

df.replace(to_replace="3+",value=4,inplace=True)
df["Total_Applicant_Income"]=df["ApplicantIncome"]+df["CoapplicantIncome"]
#drop unnecessary columns
cols=["CoapplicantIncome"]
df=df.drop(columns=cols,axis=1)
df.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
#button
if st.button("Enter"):
    prediction = rf_classifier.predict(df)
    if prediction == 1:
        st.success("Loan application is likely to be approved.")
    else:
        st.warning("Sorry, Loan application is likely to be rejected.")