import scipy
import streamlit as st
import joblib
import pandas as pd
import numpy as np


@st.cache(allow_output_mutation=True)
def load_data():
    '''
    load_data() returns a dictionary where all saved models are loaded.

    return: dict

    '''
    parameters = {'model':joblib.load('models/random_forest_regressor.sav'), 

            'sex':joblib.load('models/map_sex.sav'),

            'smoker':joblib.load('models/map_smoker.sav'),

            'region':joblib.load('models/map_region.sav'),

            'bmi':joblib.load('models/scale_bmi.sav'),

            'age_children':joblib.load('models/scale_age_children.sav'),

            'expenses':joblib.load('models/scale_target.sav')}


    return parameters


def user_input_features():
    '''
    Take user input from sidebar.

    return: pd.DataFrame Object
    '''
    inputFile = st.sidebar.file_uploader("Upload DataSet", type=['csv', 'xlsx', 'xls'])
    cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


    try:
        if inputFile is not None:
            # validing whether it a csv or excel file

            fileName = inputFile.name
            if fileName[-3:] == 'csv':
                df = pd.read_csv(inputFile)
            
            else:
                df = pd.read_excel(inputFile)

            if df.shape[1]!=len(cols) or list(df.columns) != cols:
                st.error(f"Given DataSet's columns be be {cols}\n\nTry Again with correct dataset")
            
            else:
                return df

    except Exception as e:
        st.exception(e)

    st.sidebar.header('User Input Parameters')

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
    """Before We feed data to our random forest regressor model we need to scale
    down values and need to perform box-cox and exponential transformation.

    Parameters
    ----------
    df : pd.DataFrame Object
        Test Values to predict target variable
    parameters : dict
        Containing all saved models

    return: pd.DataFrame Object
    """
    # label encoding ------------------------------------
    df['sex'] = df['sex'].map(parameters['sex'])  
    df['smoker'] = df['smoker'].map(parameters['smoker'])  
    df['region'] = df['region'].map(parameters['region'])  
    # ---------------------------------------------------

    # performing box-cox tranformation with fixed lmbda value
    lmbda_bmi=0.458601865640217
    df['bmi'] = ((df['bmi']**lmbda_bmi) -1) / lmbda_bmi
    #----------------------------------------------------

    # performing exponential tranformation
    df['age'] = df['age'] ** (1/ 1.2)
    #----------------------------------------------------

    # scaling using StandardScaler 
    df[['age', 'children']] = parameters['age_children'].transform(df[['age', 'children']])

    # scaling using robust scaler
    df['bmi'] = parameters['bmi'].transform(np.array(df['bmi']).reshape(-1, 1))

    return df


def predictTarget(X, parameters):
    """Calculate Target Variable values

    Parameters
    ----------
    df : pd.DataFrame Object or np.ndarray Object
        Test Values to predict target variable
    parameters : dict
        Containing all saved models

    return: y_pred value float type
    """
    y_pred_scaled = parameters['model'].predict(X) # scaled down value

    y_pred = parameters['expenses'].inverse_transform(y_pred_scaled) # reverse scaling

    y_pred = scipy.special.inv_boxcox(y_pred, 0.04364902969059508) # reverse box-cox 

    if y_pred.shape[0] == 1:
        return y_pred[0]
    
    else:
        return y_pred
    
