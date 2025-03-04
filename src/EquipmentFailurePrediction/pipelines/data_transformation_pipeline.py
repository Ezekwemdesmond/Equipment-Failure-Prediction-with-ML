import sys
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.components.data_transformation import DataTransformation
from EquipmentFailurePrediction.entity.config_entity import DataTransformationConfig
from EquipmentFailurePrediction.utils.common import read_yaml
from pathlib import Path

def data_transformation_pipeline(config_path: Path) -> None:
    """
    Executes the data transformation pipeline using the provided configuration file.

    Args:
        config_path (Path): Path to the data transformation configuration file.
    """
    try:
        custom_logger.info("Starting data transformation pipeline.")

        # Load the data transformation configuration
        config = read_yaml(config_path)
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config["data_transformation"]["root_dir"]),
            train_data_path=Path(config["data_transformation"]["train_data_path"]),
            test_data_path=Path(config["data_transformation"]["test_data_path"]),
            preprocessor_path=Path(config["data_transformation"]["preprocessor_path"]),
            target_column=config["data_transformation"]["target_column"],
            numerical_columns=config["data_transformation"]["numerical_columns"],
            categorical_columns=config["data_transformation"]["categorical_columns"],
        )

        # Create the data transformation object and run the process
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.run()

        custom_logger.info("Data transformation pipeline completed successfully.")
    except Exception as e:
        custom_logger.error("Error during data transformation pipeline.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_path = Path("configs/config.yaml")
        custom_logger.info(f"Executing Data Transformation Pipeline with config: {config_path}")
        data_transformation_pipeline(config_path)
    except Exception as e:
        custom_logger.error("Error in data transformation pipeline execution.")
        raise CustomException(e, sys)