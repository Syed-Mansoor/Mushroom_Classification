import os
import sys
from mushroom.components.data_ingestion import DataIngestion
from mushroom.components.data_transformation import DataTransformation
from mushroom.components.model_trainer import ModelTrainer
from mushroom.exception import MushroomException
from mushroom.logger import logging

class TrainPipeline:

    def __init__(self):
        """
        Initializes the TrainPipeline class.
        """
        try:
            self.data_ingestion = DataIngestion()  # Initialize data ingestion component
            self.data_transformation = DataTransformation()  # Initialize data transformation component
            self.model_trainer = ModelTrainer()  # Initialize model trainer component
        except Exception as e:
            raise MushroomException(e, sys)

    def initiate_data_ingestion(self):
        """
        Initiates data ingestion and returns paths of train and test datasets.
        """
        try:
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            raise MushroomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        Initiates data transformation and returns transformed data.
        """
        try:
            return self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        except Exception as e:
            raise MushroomException(e, sys)

    def initiate_model_trainer(self, train_features, train_target, test_features, test_target):
        """
        Initiates the model trainer and returns accuracy score and the trained model.
        """
        try:
            accuracy, best_model = self.model_trainer.initiate_model_trainer(train_features, train_target, test_features, test_target)
            return accuracy, best_model
        except Exception as e:
            raise MushroomException(e, sys)

    def run_pipeline(self):
        """
        Executes the full training pipeline from data ingestion to model training.
        """
        try:
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            (
                input_feature_train_arr,
                target_train_arr,
                input_feature_test_arr,
                target_test_arr,
                preprocessor_obj_file_path
            ) = self.initiate_data_transformation(train_data_path, test_data_path)

            # Step 3: Model Training
            accuracy, trained_model = self.initiate_model_trainer(
                input_feature_train_arr, 
                target_train_arr, 
                input_feature_test_arr, 
                target_test_arr
            )

            logging.info("Data ingestion, transformation, and model training completed successfully.")
            return {
                "train_features": input_feature_train_arr,
                "train_target": target_train_arr,
                "test_features": input_feature_test_arr,
                "test_target": target_test_arr,
                "preprocessor_file": preprocessor_obj_file_path,
                "trained_model": trained_model,
                "accuracy": accuracy
            }
        
        except Exception as e:
            raise MushroomException(e, sys)
