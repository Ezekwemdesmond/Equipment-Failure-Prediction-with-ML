import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from EquipmentFailurePrediction.pipelines.prediction_pipeline import prediction_pipeline


app = Flask(__name__)

# Path to the configuration file
CONFIG_PATH = Path("configs/config.yaml")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {
            'Type': request.form['type'],
            'Air temperature [K]': float(request.form['air_temperature']),
            'Process temperature [K]': float(request.form['process_temperature']),
            'Rotational speed [rpm]': float(request.form['rotational_speed']),
            'Torque [Nm]': float(request.form['torque']),
            'Tool wear [min]': float(request.form['tool_wear'])
        }
        
        # Convert form data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Run the prediction pipeline
        predictions = prediction_pipeline(CONFIG_PATH, input_df)
        
        # Get the prediction result
        prediction_result = predictions.iloc[0]
        
        return render_template('result.html', prediction=prediction_result)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)