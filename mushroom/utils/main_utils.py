import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

from mushroom.exception import MushroomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    """
    This function saves a given python object to a file using pickle.

    Parameters
    ----------
    file_path : str
        The path to the file where the object is to be saved
    obj : Object
        The object that is to be saved

    Returns
    -------
    None
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise MushroomException(e,sys)
    
def evaluate_model(TrainFeatures, TrainTarget, TestFeatures, TestTarget, models, params):
    """
    Evaluate the performance of different machine learning models and
    return a report of evaluation metrics.

    Parameters
    ----------
    TrainFeatures : array-like of shape (n_samples, n_features)
        Features of the training set
    TrainTarget : array-like of shape (n_samples,)
        Target values of the training set
    TestFeatures : array-like of shape (n_samples, n_features)
        Features of the test set
    TestTarget : array-like of shape (n_samples,)
        Target values of the test set
    models : dictionary of models
        Dictionary of machine learning models
    params : dictionary of parameters
        Dictionary of parameters to be used in GridSearchCV

    Returns
    -------
    report : dictionary
        A dictionary containing the evaluation metrics for each model
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(TrainFeatures, TrainTarget)
            
            model.set_params(**gs.best_params_)            
            model.fit(TrainFeatures, TrainTarget)
            
            y_train_pred = model.predict(TrainFeatures)
            y_test_pred = model.predict(TestFeatures)
            
            train_model_score = accuracy_score(TrainTarget, y_train_pred)
            test_model_score = accuracy_score(TestTarget, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise MushroomException(e,sys)
    
def load_object(file_path):
    """
    Description: This function is used to load the saved object from the specified file path
    
    file_path: The path from where the object is to be loaded
    
    returns: The loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise MushroomException(e,sys)
    