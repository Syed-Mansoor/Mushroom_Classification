import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from mushroom.exception import MushroomException
from mushroom.logger import logging
from mushroom.utils.main_utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'data_transformation', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        """
        This is the constructor method for DataTransformation class.
        
        It initialises the data_transformation_config attribute by creating an instance of DataTransformationConfig.
        
        It also creates the artifacts directory if it doesn't exist.
        """
    
        self.data_transformation_config = DataTransformationConfig()
        
        # Create the artifacts directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
        
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """
        try:
            columns = ['bruises', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-root', 'ring-type', 'spore-print-color']
            
            pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Feature columns: {columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("pipeline", pipeline, columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise MushroomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for data transformation and feature engineering.
        It reads the train and test data from the given paths, creates a preprocessor object, and
        applies it to the input features of the train and test data. It also applies a label encoder
        to the target data. The preprocessor object is saved to the specified file path.
        
        Args:
            train_path (str): The path to the train data.
            test_path (str): The path to the test data.
        
        Returns:
            A tuple containing the transformed input features for the train data, the target data for the train data, 
            the transformed input features for the test data, the target data for the test data, and the file path of the saved preprocessor object.
        """
        try:
            logging.info("Reading train and test data")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Obtaining preprocessor object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = "class"
            
            input_feature_train_df = train_df.drop([target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop([target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessor object to train and test input features")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("Applying label encoder to target data")
            
            le = LabelEncoder()
            
            target_train_arr = le.fit_transform(target_feature_train_df)
            target_test_arr = le.transform(target_feature_test_df)
            
            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                input_feature_train_arr,
                target_train_arr,
                input_feature_test_arr,
                target_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
                       
        except Exception as e:
            raise MushroomException(e, sys)
