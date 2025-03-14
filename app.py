from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
age_model = joblib.load('models/age_model.pkl')
gender_model = joblib.load('models/gender_model.pkl')
race_model = joblib.load('models/race_model.pkl')
marital_model = joblib.load('models/marital_model.pkl')

# Load encoders
car_encoder = joblib.load('models/car_encoder.pkl')
gender_encoder = joblib.load('models/gender_encoder.pkl')
race_encoder = joblib.load('models/race_encoder.pkl')
marital_encoder = joblib.load('models/marital_encoder.pkl')

# Get list of unique car models
car_models = sorted(car_encoder.classes_)

@app.route('/')
def home():
    return render_template('index.html', car_models=car_models)

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected car model from form
    car_model = request.form.get('car_model')
    
    # Encode the car model
    car_encoded = car_encoder.transform([car_model])[0]
    car_input = np.array([car_encoded]).reshape(1, -1)
    
    # Make predictions
    age_pred = age_model.predict(car_input)[0]
    gender_proba = gender_model.predict_proba(car_input)[0]
    race_proba = race_model.predict_proba(car_input)[0]
    marital_proba = marital_model.predict_proba(car_input)[0]
    
    # Get the class with highest probability
    gender_pred = gender_encoder.inverse_transform([gender_proba.argmax()])[0]
    race_pred = race_encoder.inverse_transform([race_proba.argmax()])[0]
    marital_pred = marital_encoder.inverse_transform([marital_proba.argmax()])[0]
    
    # Format confidence scores
    gender_confidence = {gender_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                         for i, prob in enumerate(gender_proba)}
    race_confidence = {race_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                       for i, prob in enumerate(race_proba)}
    marital_confidence = {marital_encoder.inverse_transform([i])[0]: round(prob*100, 2) 
                          for i, prob in enumerate(marital_proba)}
    
    # Prepare results for display
    results = {
        'car_model': car_model,
        'age': round(age_pred, 1),
        'gender': gender_pred,
        'gender_confidence': gender_confidence,
        'race': race_pred,
        'race_confidence': race_confidence,
        'marital_status': marital_pred,
        'marital_confidence': marital_confidence
    }
    
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
