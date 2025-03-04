import sys
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.entity.config_entity import DataTransformationConfig
from pathlib import Path
import joblib

class DataTransformation:
    """
    A class to handle data transformation, including scaling and encoding.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the DataTransformation class with the provided configuration.

        Args:
            config (DataTransformationConfig): Configuration for data transformation.
        """
        self.config = config

    def get_data_transformer(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer for scaling numerical features and encoding categorical features.

        Returns:
            ColumnTransformer: A preprocessor for the dataset.
        """
        try:
            custom_logger.info("Creating data transformer.")

            # Define the numerical and categorical columns
            numerical_columns = self.config.numerical_columns
            categorical_columns = self.config.categorical_columns

            # Create pipelines for numerical and categorical features
            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),  # Scale numerical features
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),  # Encode categorical features
                ]
            )

            # Combine the pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_columns),
                    ("cat", categorical_pipeline, categorical_columns),
                ]
            )

            custom_logger.info("Data transformer created successfully.")
            return preprocessor
        except Exception as e:
            custom_logger.error("Error creating data transformer.")
            raise CustomException(e, sys)

    def apply_transformation(self, preprocessor: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the transformation to the dataset.

        Args:
            preprocessor (ColumnTransformer): The preprocessor to apply.
            data (pd.DataFrame): The dataset to transform.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        try:
            custom_logger.info("Applying data transformation.")

            # Drop unnecessary columns
            columns_to_drop = ["Product ID", "UDI", "Failure Type"]
            data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors="ignore")

            # Separate features and target
            X = data.drop(columns=[self.config.target_column])
            y = data[self.config.target_column]

            # Apply the transformation to the features
            X_transformed = preprocessor.fit_transform(X)

            # Convert transformed features into a DataFrame
            transformed_feature_names = preprocessor.get_feature_names_out()
            transformed_data = pd.DataFrame(X_transformed, columns=transformed_feature_names)

            # Add the target column back
            transformed_data[self.config.target_column] = y.reset_index(drop=True)

            custom_logger.info("Data transformation applied successfully.")
            return transformed_data
        except Exception as e:
            custom_logger.error("Error applying data transformation.")
            raise CustomException(e, sys)

    def save_preprocessor(self, preprocessor: ColumnTransformer) -> None:
        """
        Saves the preprocessor to the specified path.

        Args:
            preprocessor (ColumnTransformer): The preprocessor to save.
        """
        try:
            custom_logger.info("Saving the preprocessor.")

            # Create the directory if it doesn't exist
            Path(os.path.dirname(self.config.preprocessor_path)).mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_path)
            custom_logger.info(f"Preprocessor saved at: {self.config.preprocessor_path}")
        except Exception as e:
            custom_logger.error("Error saving the preprocessor.")
            raise CustomException(e, sys)

    def run(self) -> None:
        """
        Executes the data transformation process.
        """
        try:
            custom_logger.info("Starting data transformation pipeline.")

            # Create the output directory if it doesn't exist
            Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)


            # Load the training and test data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            # Create the preprocessor
            preprocessor = self.get_data_transformer()

            # Apply the transformation to the training and test data
            transformed_train_data = self.apply_transformation(preprocessor, train_data)
            transformed_test_data = self.apply_transformation(preprocessor, test_data)

            # Save the transformed data
            transformed_train_data.to_csv(Path(self.config.root_dir) / "transformed_train.csv", index=False)
            transformed_test_data.to_csv(Path(self.config.root_dir) / "transformed_test.csv", index=False)

            # Save the preprocessor
            self.save_preprocessor(preprocessor)

            custom_logger.info("Data transformation pipeline completed successfully.")
        except Exception as e:
            custom_logger.error("Error during data transformation pipeline.")
            raise CustomException(e, sys)
