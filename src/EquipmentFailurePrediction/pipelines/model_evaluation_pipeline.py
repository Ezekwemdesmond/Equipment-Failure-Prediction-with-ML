import sys
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.components.model_evaluation import ModelEvaluation
from EquipmentFailurePrediction.entity.config_entity import ModelEvaluationConfig
from EquipmentFailurePrediction.utils.common import read_yaml
from pathlib import Path

def model_evaluation_pipeline(config_path: Path) -> None:
    """
    Executes the model evaluation pipeline using the provided configuration file.

    Args:
        config_path (Path): Path to the model evaluation configuration file.
    """
    try:
        custom_logger.info("Starting model evaluation pipeline.")

        # Load the model evaluation configuration
        config = read_yaml(config_path)
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config["model_evaluation"]["root_dir"]),
            test_data_path=Path(config["model_evaluation"]["test_data_path"]),
            model_path=Path(config["model_evaluation"]["model_path"]),
            target_column=config["model_evaluation"]["target_column"],
        )

        # Create the model evaluation object and run the process
        model_evaluation = ModelEvaluation(model_evaluation_config)
        model_evaluation.run()

        custom_logger.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        custom_logger.error("Error during model evaluation pipeline.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_path = Path("configs/config.yaml")
        model_evaluation_pipeline(config_path)
    except Exception as e:
        custom_logger.error("Error in model evaluation pipeline execution.")
        raise CustomException(e, sys)

