import streamlit as st
import pandas as pd
from utility import *


parameters = load_data()

st.write("""
# Insurance Premium Prediction
This app predicts the medical charges of an individual and depending upon the medical bills company can think whether to give you insurance or not.

""")

st.sidebar.header('User Input Parameters')


X = user_input_features()


st.header('User Input parameters')
st.write("""'
To run this model properly it requires some parameters, like individual's **age**, **sex**, **bmi**, number of **children** individual have, whether he/ she is a **smoker** or not and **region** where his/her house is. 

Here **medical charges** will be our dependent variable [target variable] and the rest will be our independent variables.

If sidebar is not present then click/touch on upper left **>** symbol and give inputs. You can change any parameter values and ML model will predict medical charges according to it.
""")

st.dataframe(X)

button_status = st.button('PREDICT')

st.header('Model Prediction')

if button_status:
    X_scaled = scaleDF(X, parameters)

    y_pred = predictTarget(X_scaled, parameters)
    st.write(f"Predicted Medical Expenses : 💲 {round(y_pred, 2)}")
else:
    st.write(f"Haven't Predicted Medical Expenses, Click on 'PREDICT' button")


st.header('About')
st.write('''
My name is Kaushik Pal, and i'm currently pursuing B.Tech in Computer Science. I like Machine Learning and deleloping ML apps. Also learning Deep Learning. I'm currently not pro at ML but I will be in future. 
Here is my linkedin account link https://linkedin.com/in/ikaushikpal


I used Random Forest Regressor model for this problem and optimized using GridSearchCV. Also to deploy this web-app I used streamlit library which is very easy to manage. I also have uploaded all codes to github https://github.com/ikaushikpal/Insurance_Premium_Prediction


If anyone want to conribute to this project I will gladly accept your PR. 

''')


st.header('Resource')
st.write('''
Dataset has been taken from https://www.kaggle.com/noordeen/insurance-premium-prediction

''')


st.write('## Thanks For Checking 😀 and Happy Coding 👨🏽‍💻')