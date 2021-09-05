import scipy
import streamlit as st
import joblib
import pandas as pd
import numpy as np


@st.cache(allow_output_mutation=True)
def load_data():

    parameters = {'model':joblib.load('models/random_forest_regressor.sav'), 

            'sex':joblib.load('models/map_sex.sav'),

            'smoker':joblib.load('models/map_smoker.sav'),

            'region':joblib.load('models/map_region.sav'),

            'bmi':joblib.load('models/scale_bmi.sav'),

            'age_children':joblib.load('models/scale_age_children.sav'),

            'expenses':joblib.load('models/scale_target.sav')}


    return parameters


def user_input_features():
    age = st.sidebar.number_input('Age ', min_value=18, max_value=64, value=18)

    sex = st.sidebar.selectbox('Sex', ['male', 'female'])

    bmi = st.sidebar.number_input('BMI (Body Mass Index) ', min_value=0.0, max_value=50.0, value=20.0)

    children = st.sidebar.number_input('No of Children', min_value=0, max_value=10)
    smoker = st.sidebar.selectbox('Do you smoke ?', ['No', 'Yes']).lower()
    region = st.sidebar.selectbox('Residential Area', ['North-East', 'South-East', 'North-West', 'South-West'])

    region = region.replace('-', '').lower()
    data = {
        "age":age,
        "sex":sex,
        'bmi':bmi,
        "children":children,
        "smoker":smoker,
        "region":region}

    features = pd.DataFrame(data, index=[0])
    return features


def scaleDF(df, parameters):
    df['sex'] = df['sex'].map(parameters['sex'])  
    df['smoker'] = df['smoker'].map(parameters['smoker'])  
    df['region'] = df['region'].map(parameters['region'])  

    lmbda=0.458601865640217
    df['bmi'] = ((df['bmi']**lmbda) -1) / lmbda

    df['age'] = df['age'] ** (1/ 1.2)

    df[['age', 'children']] = parameters['age_children'].transform(df[['age', 'children']])

    df['bmi'] = parameters['bmi'].transform(np.array(df['bmi']).reshape(-1, 1))

    return df


def predictTarget(X, parameters):
    y_pred_scaled = parameters['model'].predict(X)

    y_pred = parameters['expenses'].inverse_transform(y_pred_scaled)

    y_pred = scipy.special.inv_boxcox(y_pred, 0.04364902969059508)

    return y_pred[0]
