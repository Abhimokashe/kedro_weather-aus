import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def extract_training_data(df):
    """ Extracting training data from raw data and droping Date column
    Arg: Pandas DataFrame here raw data
    Return: DataFrame i.e. Training data"""
    df = df.drop('Date',axis=1)
    df1 = df[df['RainTomorrow'].notna()]
    return df1

def treat_missing_val(df1):
    """ Misssing value treatment on training data
    Arg: df1
    Return: df1_treat_missing_value"""
    df1_treat_missing_value = df1.fillna(method='ffill',axis=0).fillna(method='bfill',axis=0)
    return df1_treat_missing_value

def train_data_split(df1_treat_missing_value):
    """ Splitting training data in X,y
    Arg: df1
    Return: X_training,y_training"""
    X_training = df1_treat_missing_value.drop(['RainTomorrow'],axis=1)
    y_training = df1_treat_missing_value['RainTomorrow']
    return X_training,y_training

def label_encoding(X_training):
    """ Label encoding converting categorical variables to numerical variables
    Arg: df1_treat_missing_value
    Return: df1_label_encoder"""
    l_encoder = LabelEncoder()
    X_training['Location'] = l_encoder.fit_transform(X_training['Location'])
    X_training['WindGustDir'] = l_encoder.fit_transform(X_training['WindGustDir'])
    X_training['WindDir9am'] = l_encoder.fit_transform(X_training['WindDir9am'])
    X_training['WindDir3pm'] = l_encoder.fit_transform(X_training['WindDir3pm'])
    X_training['RainToday'] = l_encoder.fit_transform(X_training['RainToday'])
    X_training0 = X_training.copy()
    return X_training0

def scaling_data(X_training0):
    """ Scaling data is used to standardized data.
    Arg: df1_label_encoder
    Return: df1_scaled_data"""

    scaler = StandardScaler()
    df1_scaled_data = scaler.fit_transform(X_training0)
    df1_scaled_data1 = pd.DataFrame(df1_scaled_data)
    return df1_scaled_data1