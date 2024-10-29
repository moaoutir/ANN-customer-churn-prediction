import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.keras')

with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)


st.title('Customer Retention Prediction')

st.subheader('Please Provide Customer Information')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Add a button to submit the data and show results
if st.button('Predict Retention'):
    result = prediction(model, input_data)
    
    st.write(result)



def prediction(model, input_data):
    input_data_scaled = transformer.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    if prediction_proba > 0.5:
        return f"This customer shows a high likelihood of leaving: {prediction_proba}"
    else:
        return f"This customer is likely to stay: {prediction_proba"




