# Equipment Failure Prediction with Machine Learning for Predictive Maintenance

## Overview

This project predicts equipment failure using machine learning (LightGBM) and deploys the model as a web application using Flask.
It is designed for predictive maintenance, helping businesses reduce downtime and maintenance costs by identifying potential equipment failures in advance.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features

- **Machine Learning Model**: A LightGBM classifier trained on historical equipment data.
- **Web Application**: A Flask-based web app for real-time predictions.

## Technologies Used

- **Python**: Programming language for model development and web application.
- **Flask**: Web framework for building the application.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning library for model training and evaluation.

## Getting Started

### Project Structure
```bash
equipment-failure-prediction/
├── app.py                  # Flask application
├── model.py                # Script to train and save the model
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
├── templates/              # HTML templates
│   ├── index.html          # Home page
│   └── result.html         # Result page
├── static/                 # Static files (CSS, images)
│   └── style.css           # CSS for styling the web app
└── models/                 # Saved models and preprocessing pipeline
    ├── preprocessing_pipeline.pkl
    └── failure_prediction_model.pkl
```

### Prerequisites

- Python 3.x
- Pip (Python package manager)

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

1. **Run the web application**:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Enter the equipment details (e.g., type, air temperature, process temperature, rotational speed, torque, and tool wear). Click the Predict button.
4. The application will display the prediction (Failure or No Failure) and the probability of failure. A response message will suggest whether maintenance is required.

## Data

The dataset used for training the model is located in `predictive_maintenance.csv`. This dataset includes historical records of equipment performance and failure incidents.

### Data Format

- **Columns**: Describe the features included in the dataset (e.g., time, temperature, pressure, failure status).
- **Rows**: Each row represents a record of equipment performance.

## Model Training

The model training script is located in `model.py`. 
To train the model, run:
```bash
python model.py
```

## Web Application

The web application is built using Flask and allows users to:

- Upload data for prediction
- View model predictions


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
