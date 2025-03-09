# Equipment Failure Prediction with Machine Learning for Predictive Maintenance

## Overview

This project predicts equipment failure using machine learning (LightGBM) and deploys the model as a web application using Flask.
It is designed for predictive maintenance, helping businesses reduce downtime and maintenance costs by identifying potential equipment failures in advance.
The project follows a modular machine learning approach with separate components for data ingestion, transformation, model training, evaluation, and prediction.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features

- **Modular ML Architecture**: Organized components for each stage of the ML lifecycle.
- **Machine Learning Model**: A LightGBM classifier trained on historical equipment data.
- **Automated Pipeline**: Master pipeline to execute all stages sequentially.
- **Web Application**: A Flask-based web app for real-time predictions.

## Technologies Used

- **Python**: Programming language for model development and web application.
- **Flask**: Python web framework for building the application.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **LightGBM**: Gradient boosting framework for predictive modeling.

## Getting Started

### Project Structure
```bash
Equipment-Prediction-app/
├── src/
│   └── EquipmentFailurePrediction/
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_evaluation.py
│       ├── entity/
│       │   ├── __init__.py
│       │   └── config_entity.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── common.py
│       └── __init__.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── css/
│       └── styles.css
├── artifacts/
│   ├── data_ingestion/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── data_transformation/
│   │   └── preprocessor.pkl
│   └── model_trainer/
│       └── model.joblib
├── logs/
├── research/
│   └── experiment.ipynb
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── app.py

```

## Getting Started

### Prerequisites

- Python 3.x
- Pip (Python package manager)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ezekwemdesmond/Equipment-Failure-Prediction-with-ML.git
   cd Equipment-Failure-Prediction-with-ML
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Setting up the Project Structure**: To create the project structure, run:
   ```bash
   python template.py
   ```

2.  **Running the Full Pipeline**: To execute the complete ML pipeline (ingestion, transformation, training, and evaluation):
   ```bash
   python src/EquipmentFailurePrediction/master_pipeline.py
   ```
3. **Running Individual Pipelines**: You can also run individual pipelines:
   ```bash
   # Data Ingestion
    python src/EquipmentFailurePrediction/pipelines/data_ingestion_pipeline.py

    # Data Transformation
    python src/EquipmentFailurePrediction/pipelines/data_transformation_pipeline.py

    # Model Training
    python src/EquipmentFailurePrediction/pipelines/model_trainer_pipeline.py

    # Model Evaluation
    python src/EquipmentFailurePrediction/pipelines/model_evaluation_pipeline.py
   ```
4. **Running the Web Application**: To start the Flask web application:
   ```bash
   python app.py
   ```

Navigate to http://127.0.0.1:5000 in your web browser.

## Data

The dataset used for training the model contains historical records of equipment performance and failure incidents.

### Data Flow
1. Data is ingested via the data ingestion component
2. The data transformation component handles preprocessing and feature engineering
3. Transformed data is used to train the model


## Model Training

The model training process is handled by the model trainer component. The pipeline follows these steps:

1. Load transformed data
2. Configure model hyperparameters
3. Train the LightGBM classifier
4. Save the trained model

### Model Experimentation
Multiple models were experimented with during the development process:

- Logistic Regression
- Random Forest
- LightGBM

LightGBM was selected as the final model as it provided the highest performance metrics. The experimentation process can be found in the experiment.ipynb notebook in the research folder.

### Research and Development

The project includes a research folder containing experiment.ipynb, which documents the exploratory data analysis, feature engineering experiments, and model selection process. This notebook serves as a reference for the development decisions made in the final implementation.


## Web Application

The web application allows users to:

- Enter equipment details (type, temperature, rotational speed, etc.)
- Get predictions on potential failures and maintenance recommendations


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning tools.
- [Flask](https://flask.palletsprojects.com/) for web framework support.
- Dataset: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
