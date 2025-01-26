from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the preprocessing pipeline and model
preprocessing_pipeline = joblib.load('models/preprocessing_pipeline.pkl')
model = joblib.load('models/failure_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {
            'Type': request.form['type'],
            'Air temperature [K]': float(request.form['air_temperature']),
            'Process temperature [K]': float(request.form['process_temperature']),
            'Rotational speed [rpm]': float(request.form['rotational_speed']),
            'Torque [Nm]': float(request.form['torque']),
            'Tool wear [min]': float(request.form['tool_wear'])
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        input_transformed = preprocessing_pipeline.transform(input_df)

        # Make prediction
        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)[:, 1]

        # Prepare the result data
        result_data = {
            'prediction': int(prediction[0]),
            'probability': float(prediction_proba[0])
        }

        # Redirect to the result page with the prediction data
        return redirect(url_for('result', **result_data))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/result')
def result():
    # Get prediction data from query parameters
    prediction = request.args.get('prediction', type=int)
    probability = request.args.get('probability', type=float)

    # Prepare the response text
    prediction_result = "Failure" if prediction == 1 else "No Failure"
    response_text = (
        "The equipment is likely to fail. Please schedule maintenance."
        if prediction == 1
        else "The equipment is in good condition. No maintenance required."
    )

    return render_template(
        'result.html',
        prediction_text=f"Prediction: {prediction_result}, Probability: {probability:.2f}",
        response_text=response_text
    )

if __name__ == '__main__':
    app.run(debug=True)