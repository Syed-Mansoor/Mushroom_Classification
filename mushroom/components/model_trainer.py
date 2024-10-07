import os
import sys
from dataclasses import dataclass

from mushroom.exception import MushroomException
from mushroom.logger import logging
from mushroom.utils.main_utils import save_object, evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn  # Import this to log sklearn models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model_trainer', "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_features_array, train_target_array, test_features_array, test_target_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_features_array,
                train_target_array,
                test_features_array,
                test_target_array
            )
            
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost Classifier": XGBClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Ada Boost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False)                
            }
            
            params = {
                "Logistic Regression": {},
                "XGBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7],
                    "criterion": ['entropy', 'gini'],
                    "splitter": ['best', 'random'],
                    "max_features": ['sqrt', 'log2']                    
                },
                "Random Forest Classifier": {
                    "criterion": ['entropy', 'gini'],
                    "max_depth": [3, 5, 7],
                    "max_features": ['sqrt', 'log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256]                                                     
                },
                "Gradient Boosting Classifier": {                    
                    "criterion": ['friedman_mse', 'squared_error'],
                    "max_depth": [3, 5, 7],                    
                    "learning_rate": [0.1, 0.01, 0.5],
                    "max_features": ['sqrt', 'log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256]                                     
                },
                "Ada Boost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5],
                    "algorithm": ['SAMME', 'SAMME.R'],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Support Vector Classifier": {
                    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                    "gamma": ['scale', 'auto']
                },
                "CatBoost Classifier": {
                    'depth': [6, 8],
                    'learning_rate': [0.1, 0.01, 0.5],
                    'iterations': [30, 50, 100]
                }                
            }
            
            model_report: dict = evaluate_model(TrainFeatures=X_train, TrainTarget=y_train,
                                                TestFeatures=X_test, TestTarget=y_test,
                                                models=models, params=params)
            
            logging.info("Model hyperparameter tuning done")
            logging.info("Model training complete")
            
            # Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise MushroomException("No Best Model Found", sys)

            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")

            # Start an MLflow run
            with mlflow.start_run():
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_param("accuracy_score", best_model_score)
                
                # Log the best model
                mlflow.sklearn.log_model(best_model, "model")

                # Save the model locally
                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
                
                logging.info("Using the best model to predict on test data")

                predicted = best_model.predict(X_test)
                accuracy_score_result = accuracy_score(y_test, predicted)
                logging.info(f"Prediction result on test data: Accuracy Score -> {accuracy_score_result}")

                return accuracy_score_result, best_model
            
        except Exception as e:
            raise MushroomException(e, sys)
