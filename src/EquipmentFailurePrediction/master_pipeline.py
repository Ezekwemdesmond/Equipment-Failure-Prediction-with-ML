import sys
from pathlib import Path
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.pipelines.data_ingestion_pipeline import data_ingestion_pipeline
from EquipmentFailurePrediction.pipelines.data_transformation_pipeline import data_transformation_pipeline
from EquipmentFailurePrediction.pipelines.model_trainer_pipeline import model_trainer_pipeline
from EquipmentFailurePrediction.pipelines.model_evaluation_pipeline import model_evaluation_pipeline
from EquipmentFailurePrediction.utils.common import read_yaml

CONFIG_PATH = Path("configs/config.yaml")

def run_pipeline():
    try:
        custom_logger.info("Starting Equipment Failure Prediction Pipeline Execution.")

        # Step 1: Data Ingestion
        custom_logger.info("Running Data Ingestion Pipeline...")
        data_ingestion_pipeline(CONFIG_PATH)

        # Step 2: Data Transformation
        custom_logger.info("Running Data Transformation Pipeline...")
        data_transformation_pipeline(CONFIG_PATH)

        # Step 3: Model Training
        custom_logger.info("Running Model Training Pipeline...")
        training_results = model_trainer_pipeline(CONFIG_PATH)
        custom_logger.info(f"Model Training Completed: {training_results}")

        # Step 4: Model Evaluation
        custom_logger.info("Running Model Evaluation Pipeline...")
        evaluation_results = model_evaluation_pipeline(CONFIG_PATH)
        custom_logger.info(f"Model Evaluation Completed: {evaluation_results}")

        custom_logger.info("Equipment Failure Prediction Pipeline Execution Completed Successfully.")
    except Exception as e:
        custom_logger.error("Pipeline Execution Failed.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_pipeline()
