import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from mushroom.exception import MushroomException
from mushroom.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data_ingestion', 'raw')
    train_data_path: str = os.path.join('artifacts', 'data_ingestion', 'feature_store')
    
    def __post_init__(self):
        # Create directories for raw data and train/test data
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.train_data_path, exist_ok=True)
        logging.info('Created data ingestion configuration and directories')


class DataIngestion:
    def __init__(self):
        """
        Method Name: __init__
        Description: This method initializes the instance of DataIngestion class.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This method initiates the data ingestion and loads the mushroom dataset from notebooks directory
        and saves it in raw_data.csv in the raw directory. Further it splits the dataset into
        train and test sets and saves them in the train_test directory.

        Args: None

        Returns: Tuple of two strings. The first string is the path of the train data and the second string
        is the path of the test data
        """
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv("/Mushroom_classification/notebooks/mushroom_data.csv")
            logging.info("Read the dataset as dataframe")

            # Save raw data
            raw_file_path = os.path.join(self.ingestion_config.raw_data_path, 'raw_data.csv')
            df.to_csv(raw_file_path, index=False, header=True)
            logging.info('Raw data saved successfully')

            # Split the dataset
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
            logging.info("Train test split completed")

            # Save train and test sets
            train_file_path = os.path.join(self.ingestion_config.train_data_path, 'train.csv')
            test_file_path = os.path.join(self.ingestion_config.train_data_path, 'test.csv')
            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)
            logging.info('Train and test sets saved successfully')

            return (
                train_file_path,
                test_file_path
            )
        
        except Exception as e:
            raise MushroomException(e, sys)
