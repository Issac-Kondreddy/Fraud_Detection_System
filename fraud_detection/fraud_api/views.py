from django.shortcuts import render
import numpy as np
import joblib
from keras.models import load_model
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import csv
import json
import logging
from django.conf import settings
import os

logger = logging.getLogger(__name__)

# Load the GBDT model, Neural Network model, and scaler once at startup
try:
    # Paths to your models
    gbdt_model_path = os.path.join(settings.MODELS_DIR, 'calibrated_gbdt_model.pkl')
    nn_model_path = os.path.join(settings.MODELS_DIR, 'nn_model.keras')
    scaler_path = os.path.join(settings.MODELS_DIR, 'scaler.pkl')

    # Load the models and scaler
    gbdt_model = joblib.load(gbdt_model_path)
    nn_model = load_model(nn_model_path)
    scaler = joblib.load(scaler_path)

    logger.info('Models and scaler loaded successfully.')

except Exception as e:
    logger.error(f"Error loading models: {e}")
    scaler = None
# Fraud detection threshold (set in settings.py or default)
FRAUD_THRESHOLD = getattr(settings, 'FRAUD_THRESHOLD', 0.39)

# Render the frontend HTML page
def index(request):
    return render(request, 'fraud_api/index.html')

# Single transaction fraud prediction
@csrf_exempt
def predict_fraud(request):
    if request.method == 'POST':
        try:
            input_data = json.loads(request.body.decode('utf-8')).get('data')
            input_array = np.array(input_data, dtype=np.float64).reshape(1, -1)

            # Scale input data
            processed_data = scaler.transform(input_array)

            # GBDT model predictions
            gbdt_preds = gbdt_model.predict_proba(processed_data)[:, 1].reshape(-1, 1)

            # Combine for NN model
            combined_input = np.hstack((processed_data, gbdt_preds))
            nn_preds = nn_model.predict(combined_input).ravel()

            # Final prediction with weighted combination
            final_pred_prob = 0.7 * gbdt_preds + 0.3 * nn_preds

            fraud_prediction = (final_pred_prob >= FRAUD_THRESHOLD).astype(int)

            logger.info(f"Fraud Probability: {final_pred_prob[0]}")

            return JsonResponse({'prediction': int(fraud_prediction[0]), 'fraud_probability': float(final_pred_prob[0])})

        except Exception as e:
            logger.error(f"Error in single transaction prediction: {e}")
            return JsonResponse({'error': 'An error occurred: ' + str(e)}, status=500)

# Batch transaction fraud prediction
@csrf_exempt
def predict_batch(request):
    if request.method == 'POST':
        try:
            if 'file' not in request.FILES:
                return JsonResponse({'error': 'File not provided'}, status=400)

            # Check if the scaler is loaded
            if scaler is None:
                logger.error('Scaler not loaded.')
                return JsonResponse({'error': 'Scaler is not loaded. Cannot process input.'}, status=500)

            file = request.FILES['file']
            file_path = default_storage.save(file.name, file)

            transactions = []
            predictions = []
            probabilities = []

            # Read and process the CSV file
            with open(file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header
                for row in reader:
                    transactions.append([float(x) for x in row])

            for transaction in transactions:
                # Scale the data using the scaler
                processed_data = scaler.transform([transaction])

                # GBDT model prediction
                gbdt_preds = gbdt_model.predict_proba(processed_data)[:, 1].reshape(-1, 1)

                # Combine for NN input
                combined_input = np.hstack((processed_data, gbdt_preds))

                # NN model prediction
                nn_preds = nn_model.predict(combined_input).ravel()

                # Final prediction
                final_pred_prob = (gbdt_preds + nn_preds) / 2
                fraud_prediction = (final_pred_prob >= FRAUD_THRESHOLD).astype(int)

                # Append predictions and probabilities
                predictions.append(int(fraud_prediction[0]))
                probabilities.append(float(final_pred_prob[0]))

            # Return predictions and probabilities
            return JsonResponse({'predictions': predictions, 'probabilities': probabilities})

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return JsonResponse({'error': 'An error occurred: ' + str(e)}, status=500)
