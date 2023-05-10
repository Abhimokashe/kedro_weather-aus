import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def extracting_inference_data(df):
    """ Extracting inference data
    Arg: df
    return: df_inf"""
    df = df.drop('Date',axis=1)
    df_inf = df[df['RainTomorrow'].isna()]
    return df_inf

def splitting_inference_data(df_inf):
    """ Splitting inference data
    Arg: df2
    Return: X_inf,y_inf """
    X_inf = df_inf.drop(['RainTomorrow'],axis=1)
    y_inf = df_inf['RainTomorrow']
    return X_inf,y_inf

def inference_data_treat_missing_val(X_inf):
    """ Misssing value treatment on training data
    Arg: df1
    Return: df1_treat_missing_value"""
    Xinf_treat_missing_value = X_inf.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)
    return Xinf_treat_missing_value

def inference_data_label_encoding(Xinf_treat_missing_value):
    """ Label encoding converting categorical variables to numerical variables
    Arg: df1_treat_missing_value
    Return: df1_label_encoder"""
    from sklearn.preprocessing import LabelEncoder
    l_encoder = LabelEncoder()
    Xinf_treat_missing_value['Location'] = l_encoder.fit_transform(Xinf_treat_missing_value['Location'])
    Xinf_treat_missing_value['WindGustDir'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindGustDir'])
    Xinf_treat_missing_value['WindDir9am'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindDir9am'])
    Xinf_treat_missing_value['WindDir3pm'] = l_encoder.fit_transform(Xinf_treat_missing_value['WindDir3pm'])
    Xinf_treat_missing_value['RainToday'] = l_encoder.fit_transform(Xinf_treat_missing_value['RainToday'])
    Xinf_treat_missing_value0 = Xinf_treat_missing_value.copy()
    return Xinf_treat_missing_value0

def inference_data_scaling(Xinf_treat_missing_value0):
    """ Scaling data is used to standardized data.
    Arg: df1_label_encoder
    Return: df1_scaled_data"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    inference_scaled_data = scaler.fit_transform(Xinf_treat_missing_value0)
    df1_inference_scaled_data = pd.DataFrame(inference_scaled_data)
    return df1_inference_scaled_data

def logregAlgorithm1(inference_scaled_data:pd.DataFrame,model) -> pd.DataFrame:
    """Logistic regression algorithm used
    Args: X_train,X_test,y_train,y_test
    Return:y_pred_train,y_pred_test"""
    # from sklearn.linear_model import LogisticRegression
    # logreg = LogisticRegression(solver='liblinear',random_state=0)
    # logreg.fit(X_train,y_train)
    """# import pickle   # This code works.
    # with open("model.pkl",'rb') as f1:
    #  model1 = pickle.load(f1)"""
     
    #model = joblib.load('data\06_models\model.pkl','rb')
    # import os
    # location = 'C:\Work\Weather_Aus\weather-aus\data\06_models'
    # fullpath = os.path.join(location, 'model.pkl')
    # model = joblib.load(fullpath)
    
    y_pred_inf = model.predict(inference_scaled_data)
    df1_y_pred_inf = pd.DataFrame(y_pred_inf)
    return df1_y_pred_inf

def concat_inf(X_inf,df1_y_pred_inf):
    """ Concating independent variable inference to predicted dependent variable inference
    Args: X_inf,df1_y_pred_inf
    Return: df_inference"""

    df_inference = pd.concat([X_inf,df1_y_pred_inf],axis=1)
    return df_inference

def concat_original(df1,df_inference):
    """ Concating df_inference to df1
    Args : df1,df_inference
    Return: df_req"""

    df_req = pd.concat([df1,df_inference],axis=0)
    return df_req