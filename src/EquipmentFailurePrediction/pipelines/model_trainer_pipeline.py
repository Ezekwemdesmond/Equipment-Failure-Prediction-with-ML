import sys
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.components.model_trainer import ModelTrainer
from EquipmentFailurePrediction.entity.config_entity import ModelTrainerConfig
from EquipmentFailurePrediction.utils.common import read_yaml
from pathlib import Path

def model_trainer_pipeline(config_path: Path) -> dict:
    """
    Executes the model training pipeline using the provided configuration file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        dict: Information about the trained model.
    """
    try:
        custom_logger.info(f"Starting model training pipeline with config from {config_path}")

        # Load the configuration
        config = read_yaml(config_path)
        model_trainer_section = config["model_trainer"]
        
        # Create model trainer config
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(model_trainer_section["root_dir"]),
            train_data_path=Path(model_trainer_section["train_data_path"]),
            test_data_path=Path(model_trainer_section["test_data_path"]),
            model_path=Path(model_trainer_section["model_path"]),
            target_column=model_trainer_section["target_column"],
            random_state=model_trainer_section["random_state"],
            test_size=model_trainer_section["test_size"],
            evaluation_metric=model_trainer_section["evaluation_metric"],
            params_file=Path(model_trainer_section.get("params_file", ""))
        )

        # Create the model trainer object and run the process
        model_trainer = ModelTrainer(model_trainer_config)
        results = model_trainer.run()

        custom_logger.info("Model training pipeline completed successfully.")
        return results
    except Exception as e:
        custom_logger.error(f"Error during model training pipeline: {e}")
        raise CustomException(e, sys)    

if __name__ == "__main__":
    try:
        config_path = Path("configs/config.yaml")
        model_trainer_pipeline(config_path)
    except Exception as e:
        custom_logger.error("Error in model training pipeline execution.")
        raise CustomException(e, sys)