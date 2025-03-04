import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.entity.config_entity import ModelTrainerConfig
from EquipmentFailurePrediction.utils.common import read_yaml
from pathlib import Path
import joblib
import os
import re

class ModelTrainer:
    """
    A class to handle model training, including SMOTE and cross-validation.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with the provided configuration.

        Args:
            config (ModelTrainerConfig): Configuration for model training.
        """
        self.config = config
        
        # Load model parameters if path is provided but parameters are not
        if self.config.params_file and not self.config.model_params:
            try:
                self.config.model_params = read_yaml(self.config.params_file)
                custom_logger.info(f"Model parameters loaded from {self.config.params_file}")
            except Exception as e:
                custom_logger.warning(f"Could not load parameters from {self.config.params_file}. Using default parameters.")
                self.config.model_params = {}
        elif not self.config.model_params:
            custom_logger.info("No model parameters provided. Using default parameters.")
            self.config.model_params = {}

    def _clean_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature names to ensure they're compatible with LightGBM.
        
        Args:
            X (pd.DataFrame): DataFrame with features
            
        Returns:
            pd.DataFrame: DataFrame with cleaned feature names
        """
        # Create a copy to avoid modifying the original DataFrame
        X_cleaned = X.copy()
        
        # Define a function to clean column names
        def clean_column_name(name):
            # Replace special JSON characters with underscores
            # This includes: " { } [ ] , : \
            cleaned_name = re.sub(r'["\{\}\[\],:\\]', '_', str(name))
            return cleaned_name
        
        # Create a mapping of old to new column names
        column_mapping = {col: clean_column_name(col) for col in X_cleaned.columns}
        
        # Rename the columns
        X_cleaned.rename(columns=column_mapping, inplace=True)
        
        # Log the renamed columns
        renamed_cols = [f"{old} -> {new}" for old, new in column_mapping.items() if old != new]
        if renamed_cols:
            custom_logger.info(f"Renamed {len(renamed_cols)} columns to be compatible with LightGBM: {', '.join(renamed_cols)}")
        
        return X_cleaned

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Trains multiple models using SMOTE and cross-validation.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            dict: A dictionary containing the best model and its evaluation score.
        """
        try:
            custom_logger.info("Starting model training with SMOTE and cross-validation.")
            
            # Clean feature names to be compatible with LightGBM
            X_train_cleaned = self._clean_feature_names(X_train)
            
            # Get model parameters from config
            model_params = self.config.model_params
            
            # Define models to train with parameters from config
            models = {
                "LogisticRegression": LogisticRegression(
                    random_state=self.config.random_state,
                    **model_params.get("LogisticRegression", {})
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    random_state=self.config.random_state,
                    **model_params.get("RandomForestClassifier", {})
                ),
                "LightGBM": LGBMClassifier(
                    random_state=self.config.random_state,
                    **model_params.get("LGBMClassifier", {})
                ),
            }

            # Initialize SMOTE for handling class imbalance with parameters from config
            smote_params = model_params.get("SMOTE", {})
            smote = SMOTE(random_state=self.config.random_state, **smote_params)

            # Evaluate models using cross-validation
            best_score = -1
            best_model = None
            best_model_name = None

            for model_name, model in models.items():
                custom_logger.info(f"Training {model_name} model with parameters: {model.get_params()}")

                try:
                    # Create a pipeline with SMOTE and the model
                    pipeline = ImbPipeline(steps=[
                        ("smote", smote),  # Apply SMOTE
                        ("model", model),  # Train the model
                    ])

                    # Perform cross-validation
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)
                    scores = cross_val_score(
                        pipeline,
                        X_train_cleaned,
                        y_train,
                        cv=cv,
                        scoring=self.config.evaluation_metric,
                        n_jobs=-1,
                    )

                    # Calculate the mean cross-validation score
                    mean_score = scores.mean()
                    custom_logger.info(f"{model_name} {self.config.evaluation_metric} score: {mean_score}")

                    # Update the best model if the current model performs better
                    if mean_score > best_score:
                        best_score = mean_score
                        
                        # Train the final model on the full dataset with SMOTE
                        final_pipeline = ImbPipeline(steps=[
                            ("smote", smote),
                            ("model", model)
                        ])
                        final_pipeline.fit(X_train_cleaned, y_train)
                        best_model = final_pipeline
                        best_model_name = model_name
                
                except Exception as e:
                    custom_logger.error(f"Error training {model_name}: {e}")
                    custom_logger.info(f"Skipping {model_name} due to training error.")
                    continue

            if best_model is None:
                raise Exception("All models failed to train. Please check your data and parameters.")

            custom_logger.info(f"Best model: {best_model_name} with {self.config.evaluation_metric} score: {best_score}")
            return {
                "best_model": best_model, 
                "best_score": best_score, 
                "best_model_name": best_model_name,
                "feature_names": X_train_cleaned.columns.tolist()  # Store cleaned feature names
            }
        except Exception as e:
            custom_logger.error(f"Error during model training: {e}")
            raise CustomException(e, sys)

    def save_model(self, model, model_name: str = None, feature_names: list = None) -> None:
        """
        Saves the trained model to the specified path.

        Args:
            model: The trained model.
            model_name: Name of the model (optional).
            feature_names: List of feature names used in training (optional).
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            
            custom_logger.info("Saving the best model.")
            joblib.dump(model, self.config.model_path)
            
            # Save model metadata if name is provided
            if model_name:
                metadata_path = os.path.join(
                    os.path.dirname(self.config.model_path), 
                    "model_metadata.txt"
                )
                with open(metadata_path, "w") as f:
                    f.write(f"Best Model: {model_name}\n")
                    f.write(f"Evaluation Metric: {self.config.evaluation_metric}\n")
                    if hasattr(model, "steps") and len(model.steps) > 1:
                        f.write("Pipeline Steps:\n")
                        for step_name, _ in model.steps:
                            f.write(f"- {step_name}\n")
                    
                    # Add feature names to metadata if provided
                    if feature_names:
                        f.write("\nFeature Names:\n")
                        for name in feature_names:
                            f.write(f"- {name}\n")
                
            custom_logger.info(f"Best model saved at: {self.config.model_path}")
        except Exception as e:
            custom_logger.error(f"Error saving the model: {e}")
            raise CustomException(e, sys)

    def run(self) -> None:
        """
        Executes the model training process.
        """
        try:
            custom_logger.info("Starting model training pipeline.")

            # Create root directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Load the training data
            train_data = pd.read_csv(self.config.train_data_path)
            custom_logger.info(f"Loaded training data from {self.config.train_data_path}")
            
            X_train = train_data.drop(columns=[self.config.target_column])
            y_train = train_data[self.config.target_column]
            
            custom_logger.info(f"Training data shape: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")

            # Train models and select the best one
            training_results = self.train_models(X_train, y_train)
            best_model = training_results["best_model"]
            best_model_name = training_results["best_model_name"]
            best_score = training_results["best_score"]
            feature_names = training_results.get("feature_names")
            
            # Log the best model name and score
            custom_logger.info(f"Selected model for deployment: {best_model_name}")
            custom_logger.info(f"Best model score ({self.config.evaluation_metric}): {best_score}")

            # Save the best model
            self.save_model(best_model, best_model_name, feature_names)

            custom_logger.info("Model training pipeline completed successfully.")
            
            return {
                "model_name": best_model_name,
                "model_score": best_score,
                "model_path": self.config.model_path
            }
        except Exception as e:
            custom_logger.error(f"Error during model training pipeline: {e}")
            raise CustomException(e, sys)