import sys
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.components.data_ingestion import DataIngestion
from EquipmentFailurePrediction.entity.config_entity import DataIngestionConfig
from EquipmentFailurePrediction.utils.common import read_yaml
from pathlib import Path

def data_ingestion_pipeline(config_path: Path) -> None:
    """
    Executes the data ingestion pipeline using the provided configuration file.

    Args:
        config_path (Path): Path to the data ingestion configuration file.
    """
    try:
        custom_logger.info("Starting data ingestion pipeline.")

        # Load the data ingestion configuration
        config = read_yaml(config_path)
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config["data_ingestion"]["root_dir"]),
            source_data_path=Path(config["data_ingestion"]["source_data_path"]),
            train_data_path=Path(config["data_ingestion"]["train_data_path"]),
            test_data_path=Path(config["data_ingestion"]["test_data_path"]),
            test_size=config["data_ingestion"]["test_size"],
            random_state=config["data_ingestion"]["random_state"],
        )

        # Create the data ingestion object and run the process
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.run()

        custom_logger.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        custom_logger.error("Error during data ingestion pipeline.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_path = Path("configs/config.yaml")
        data_ingestion_pipeline(config_path)
    except Exception as e:
        custom_logger.error("Error in data ingestion pipeline execution.")
        raise CustomException(e, sys)