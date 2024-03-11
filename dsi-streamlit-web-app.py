
#####################################

# CODING THE WEB APP

#####################################

# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import joblib


# LOAD MODEL PIPELINE
model = joblib.load("model.joblib")


# ADD TITLE AND INSTRUCTIONS
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit likelihood to purchase")


#####################################

# BUILDING THE FORMS

#####################################



# AGE INPUT FORM
age = st.number_input(
    label = "01. Enter the Customer's Age",
    min_value = 18,
    max_value = 120,
    value = 35)


# GENDER INPUT FORM
gender = st.radio(
    label = "02. Enter the Customer's Gender",
    options = ['M','F'])


# CREDIT SCORE INPUT FORM
credit_score = st.number_input(
    label = "03. Enter the Customer's Credit Score",
    min_value = 0,
    max_value = 1000,
    value = 500)

# SUBMIT INPUTS TO MODEL
if st.button("Submit for Prediction"):
    
    # STORE DATA IN DF FOR PREDICTION
    new_data = pd.DataFrame({"age" : [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # APPLY MODEL PIPELINE TO PINPUT DATA AND EXTRACT PROBABILITY PREDICTION
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # OUTPUT PREDICTION
    st.subheader(f"Basesd on these customer attributed, our mdodel predicts a purchase probability of {pred_proba:.0%}")

    
    
    
    