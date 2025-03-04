import sys
from pathlib import Path
import pandas as pd
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.components.prediction import PredictionPipeline
from EquipmentFailurePrediction.utils.common import read_yaml

def prediction_pipeline(config_path: Path, input_data: pd.DataFrame) -> pd.Series:
    """
    Executes the prediction pipeline using the provided configuration file and input data.

    Args:
        config_path (Path): Path to the configuration file.
        input_data (pd.DataFrame): The input data for prediction.

    Returns:
        pd.Series: The predicted values as human-readable text.
    """
    try:
        custom_logger.info("Starting prediction pipeline.")

        # Load the configuration
        config = read_yaml(config_path)

        # Define paths to the model and preprocessor
        model_path = Path(config["model_trainer"]["model_path"])
        preprocessor_path = Path(config["data_transformation"]["preprocessor_path"])

        # Initialize the prediction pipeline
        prediction_pipeline = PredictionPipeline(model_path, preprocessor_path)

        # Run the prediction pipeline
        predictions = prediction_pipeline.run(input_data)

        custom_logger.info("Prediction pipeline completed successfully.")
        return predictions
    except Exception as e:
        custom_logger.error("Error during prediction pipeline.")
        raise CustomException(e, sys)