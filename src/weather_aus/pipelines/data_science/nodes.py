import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_test_splitting(df1_scaled_data1,y_training):
    """ data split into training and testing
    Args: df1_scaled_data,y_training
    Return: X_train,X_test,y_train,y_test"""
    
    X_train,X_test,y_train,y_test = train_test_split(df1_scaled_data1,y_training,random_state=0,test_size=0.20)
    return X_train,X_test,y_train,y_test

def logregAlgorithm(X_train,X_test,y_train,y_test):
    """Logistic regression algorithm used
    Args: X_train,X_test,y_train,y_test
    Return:y_pred_train,y_pred_test"""
    
    logreg = LogisticRegression(solver='liblinear',random_state=0)
    model = logreg.fit(X_train,y_train)
    """# import pickle   # This code works but we don't use.
    # # filename = 'model'
    # # outfile = open('filename','wb')
    # with open('model_pkl','wb') as f:
    #  pickle.dump(model,f)"""

    return model



def prediction(X_train,X_test,y_train,y_test,model):
    y_pred_train = model.predict(X_train)
    y_pred_train = pd.Series(y_pred_train)
    y_pred_test = model.predict(X_test)
    y_pred_test = pd.Series(y_pred_test)
    return (y_pred_train,y_pred_test)

def evaluation(y_pred_train:pd.Series,y_pred_test:pd.Series,y_train:pd.Series,y_test:pd.Series):

    """Model accuracy check
    Args: y_pred_train,y_pred_test
    Return: Accuracy_score"""
    
    accuracy_score_train = accuracy_score(y_train,y_pred_train)
    accuracy_score_train = pd.Series(accuracy_score_train)
    accuracy_score_test = accuracy_score(y_test,y_pred_test)
    accuracy_score_test = pd.Series(accuracy_score_test)
    return (accuracy_score_train,accuracy_score_test)