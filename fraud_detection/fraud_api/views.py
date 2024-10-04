from django.shortcuts import render
import numpy as np
import joblib
from keras.models import load_model
from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler
from django.views.decorators.csrf import csrf_exempt
import json

# Load the GBDT model and Neural Network model
gbdt_model = joblib.load('/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/models/gbdt_model.pkl')  # Replace with actual path
nn_model = load_model('/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/models/nn_model.keras')    # Replace with actual path
scaler = joblib.load('/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/models/scaler.pkl')     
# Preprocessing function (if necessary)
def preprocess_input(input_data):
    scaler = StandardScaler()
    return scaler.transform(input_data)

# Prediction function
@csrf_exempt
def predict_fraud(request):
    if request.method == 'POST':
        try:
            # Parse the input data as JSON
            input_data = json.loads(request.body.decode('utf-8')).get('data')

            # Convert the input data to a NumPy array of floats
            input_array = np.array(input_data, dtype=np.float64)

            # Ensure input is reshaped correctly (1, -1) for a single sample
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)

            # Scale the input data using the saved scaler
            processed_data = scaler.transform(input_array)

            # GBDT model prediction (probabilities)
            gbdt_preds = gbdt_model.predict_proba(processed_data)[:, 1].reshape(-1, 1)

            # Combine GBDT output with original scaled features for NN input
            combined_input = np.hstack((processed_data, gbdt_preds))

            # Neural Network prediction (probabilities)
            nn_preds = nn_model.predict(combined_input).ravel()

            # Print or log the probability of fraud
            print(f"Fraud Probability: {nn_preds[0]}")  # Log the probability

            # Convert the prediction to binary based on threshold (0.5 by default)
            fraud_prediction = (nn_preds >= 0.3).astype(int)

            # Return the result as a JSON response
            return JsonResponse({'prediction': int(fraud_prediction), 'fraud_probability': float(nn_preds[0])})

        except Exception as e:
            return JsonResponse({'error': 'An error occurred: ' + str(e)}, status=500)
