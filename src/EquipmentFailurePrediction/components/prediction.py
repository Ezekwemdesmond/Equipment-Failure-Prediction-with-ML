import sys
import pandas as pd
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from pathlib import Path
import joblib

class PredictionPipeline:
    """
    A class to handle making predictions using the trained model.
    """

    def __init__(self, model_path: Path, preprocessor_path: Path):
        """
        Initializes the PredictionPipeline class with the provided model and preprocessor paths.

        Args:
            model_path (Path): Path to the trained model.
            preprocessor_path (Path): Path to the preprocessor.
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def load_model_and_preprocessor(self):
        """
        Loads the trained model and preprocessor.

        Returns:
            tuple: The loaded model and preprocessor.
        """
        try:
            custom_logger.info("Loading the trained model and preprocessor.")
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            custom_logger.info("Model and preprocessor loaded successfully.")
            return model, preprocessor
        except Exception as e:
            custom_logger.error("Error loading the model or preprocessor.")
            raise CustomException(e, sys)

    def preprocess_data(self, preprocessor, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input data using the loaded preprocessor.

        Args:
            preprocessor: The loaded preprocessor.
            data (pd.DataFrame): The input data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        try:
            custom_logger.info("Preprocessing the input data.")
            preprocessed_data = preprocessor.transform(data)
            custom_logger.info("Input data preprocessed successfully.")
            return preprocessed_data
        except Exception as e:
            custom_logger.error("Error preprocessing the input data.")
            raise CustomException(e, sys)

    def predict(self, model, preprocessed_data: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained model and maps them to human-readable text.

        Args:
            model: The trained model.
            preprocessed_data (pd.DataFrame): The preprocessed input data.

        Returns:
            pd.Series: The predicted values as human-readable text.
        """
        try:
            custom_logger.info("Making predictions.")
            predictions = model.predict(preprocessed_data)

            # Map predictions to human-readable text
            prediction_mapping = {0: "No Failure", 1: "Failure"}
            predictions_text = pd.Series(predictions).map(prediction_mapping)

            custom_logger.info("Predictions made successfully.")
            return predictions_text
        except Exception as e:
            custom_logger.error("Error making predictions.")
            raise CustomException(e, sys)

    def run(self, input_data: pd.DataFrame) -> pd.Series:
        """
        Executes the prediction pipeline.

        Args:
            input_data (pd.DataFrame): The input data for prediction.

        Returns:
            pd.Series: The predicted values as human-readable text.
        """
        try:
            custom_logger.info("Starting prediction pipeline.")

            # Load the model and preprocessor
            model, preprocessor = self.load_model_and_preprocessor()

            # Preprocess the input data
            preprocessed_data = self.preprocess_data(preprocessor, input_data)

            # Make predictions
            predictions = self.predict(model, preprocessed_data)

            custom_logger.info("Prediction pipeline completed successfully.")
            return predictions
        except Exception as e:
            custom_logger.error("Error during prediction pipeline.")
            raise CustomException(e, sys)

