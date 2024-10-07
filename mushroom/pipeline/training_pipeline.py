import os
import sys
from mushroom.components.data_ingestion import DataIngestion
from mushroom.components.data_transformation import DataTransformation
from mushroom.exception import MushroomException
from mushroom.logger import logging

class TrainPipeline:

    def __init__(self):
        """
        This is the constructor method for TrainPipeline class.

        It initializes the data ingestion component.

        Raises:
            MushroomException: If there is an error during instantiation of DataIngestion class.
        """

        try:
            self.data_ingestion = DataIngestion()  # Initialize data ingestion component
        except Exception as e:
            raise MushroomException(e, sys)
        
    def initiate_data_ingestion(self):
        """
        This method initiates the data ingestion and returns the path of the train and test data sets.

        Returns:
            tuple: A tuple containing the path of the train data set and the path of the test data set.
        """

        try:
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path
        except Exception as e:
            raise MushroomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        """
        This method initiates the data transformation and feature engineering, and returns the transformed input features of the train and test data sets, the target data of the train and test data sets, and the file path of the saved preprocessor object.

        Args:
            train_data_path (str): The path of the train data set.
            test_data_path (str): The path of the test data set.

        Returns:
            tuple: A tuple containing the transformed input features of the train data set, the target data of the train data set, the transformed input features of the test data set, the target data of the test data set, and the file path of the saved preprocessor object.
        """
        try:
            data_transformation = DataTransformation()
            return data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        except Exception as e:
            raise MushroomException(e, sys)

    def run_pipeline(self):
        """
        This method runs the training pipeline. It first performs data ingestion and returns the path of the train and test data sets. Then, it performs data transformation and feature engineering, and returns the transformed input features of the train and test data sets, the target data of the train and test data sets, and the file path of the saved preprocessor object.

        Returns:
            dict: A dictionary containing the transformed input features of the train data set, the target data of the train data set, the transformed input features of the test data set, the target data of the test data set, and the file path of the saved preprocessor object.

        Raises:
            MushroomException: If there is an error during the execution of the training pipeline.
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

            logging.info("Data ingestion and transformation completed successfully.")
            return {
                "train_features": input_feature_train_arr,
                "train_target": target_train_arr,
                "test_features": input_feature_test_arr,
                "test_target": target_test_arr,
                "preprocessor_file": preprocessor_obj_file_path
            }
        except Exception as e:
            raise MushroomException(e, sys)
