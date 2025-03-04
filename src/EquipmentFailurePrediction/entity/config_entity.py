from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    """
    root_dir: Path
    source_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    """
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    target_column: str
    numerical_columns: list
    categorical_columns: list

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training.
    """
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    target_column: str
    random_state: int
    test_size: float
    evaluation_metric: str
    params_file: Path = None
    model_params: dict = None

@dataclass
class ModelEvaluationConfig:
    """
    Configuration class for model evaluation.
    """
    root_dir: Path
    test_data_path: Path
    model_path: Path
    target_column: str