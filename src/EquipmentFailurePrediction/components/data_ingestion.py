import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    """
    A class to handle data ingestion, including loading and splitting the dataset.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            custom_logger.info("Loading dataset from CSV file.")
            data = pd.read_csv(self.config.source_data_path)
            custom_logger.info("Dataset loaded successfully.")
            return data
        except Exception as e:
            custom_logger.error("Error loading dataset.")
            raise CustomException(e, sys)

    def split_data(self, data: pd.DataFrame) -> None:
        """
        Splits the dataset into training and testing sets and saves them to the specified paths.

        Args:
            data (pd.DataFrame): The dataset to split.
        """
        try:
            custom_logger.info("Splitting dataset into training and testing sets.")
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            custom_logger.info("Dataset split successfully.")

            # Save the training and testing datasets
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)
            custom_logger.info("Training and testing datasets saved.")
        except Exception as e:
            custom_logger.error("Error splitting dataset.")
            raise CustomException(e, sys)

    def run(self) -> None:
        """
        Executes the data ingestion process, including loading and splitting the dataset.
        """
        try:
            custom_logger.info("Starting data ingestion.")

            # Create the output directory if it doesn't exist
            Path(os.path.dirname(self.config.train_data_path)).mkdir(parents=True, exist_ok=True)
            
            data = self.load_data()
            self.split_data(data)
            custom_logger.info("Data ingestion completed successfully.")
        except Exception as e:
            custom_logger.error("Error during data ingestion.")
            raise CustomException(e, sys)