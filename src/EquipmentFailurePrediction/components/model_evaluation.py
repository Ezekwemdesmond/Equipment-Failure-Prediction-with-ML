import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import json
from EquipmentFailurePrediction.exception import CustomException
from EquipmentFailurePrediction.logger import logger as custom_logger
from EquipmentFailurePrediction.entity.config_entity import ModelEvaluationConfig
import joblib

class ModelEvaluation:
    """
    A class to handle model evaluation.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation class with the provided configuration.

        Args:
            config (ModelEvaluationConfig): Configuration for model evaluation.
        """
        self.config = config

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model on the test data using multiple metrics.

        Args:
            model: The trained model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        try:
            custom_logger.info("Evaluating the model on the test data.")

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON serialization

            # Create a dictionary of evaluation results
            evaluation_results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc_score": roc_auc,
                "confusion_matrix": conf_matrix,
            }

            custom_logger.info("Model evaluation completed successfully.")
            return evaluation_results
        except Exception as e:
            custom_logger.error("Error evaluating the model.")
            raise CustomException(e, sys)

    def save_evaluation_results(self, evaluation_results: dict) -> None:
        """
        Saves the evaluation results as a JSON file.

        Args:
            evaluation_results (dict): A dictionary containing evaluation metrics.
        """
        try:
            # Create the evaluation directory if it doesn't exist
            evaluation_dir = self.config.root_dir
            evaluation_dir.mkdir(parents=True, exist_ok=True)

            # Save the results as a JSON file
            evaluation_filepath = evaluation_dir / "evaluation_results.json"
            with open(evaluation_filepath, "w") as f:
                json.dump(evaluation_results, f, indent=4)
            custom_logger.info(f"Evaluation results saved at: {evaluation_filepath}")
        except Exception as e:
            custom_logger.error("Error saving evaluation results.")
            raise CustomException(e, sys)

    def run(self) -> None:
        """
        Executes the model evaluation process.
        """
        try:
            custom_logger.info("Starting model evaluation pipeline.")

            # Load the test data
            test_data = pd.read_csv(self.config.test_data_path)
            X_test = test_data.drop(columns=[self.config.target_column])
            y_test = test_data[self.config.target_column]

            # Load the trained model
            model = joblib.load(self.config.model_path)

            # Evaluate the model
            evaluation_results = self.evaluate_model(model, X_test, y_test)

            # Save the evaluation results
            self.save_evaluation_results(evaluation_results)

            custom_logger.info("Model evaluation pipeline completed successfully.")
        except Exception as e:
            custom_logger.error("Error during model evaluation pipeline.")
            raise CustomException(e, sys)
