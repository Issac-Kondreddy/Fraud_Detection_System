
# Fraud Detection System

## Overview
This project is a **fraud detection system** an internal tool that combines a **LightGBM** (Gradient Boosting Decision Trees) model with a **Neural Network** for enhanced performance. The system uses a hybrid approach to predict fraudulent transactions, and the models are containerized using **Docker** for easy deployment. Additionally, **LIME** is used for explainability, helping to understand which features are most important in predicting fraud.

## Features
- **Hybrid Model**: Combines a **LightGBM** model with a **Neural Network**.
- **LIME Explainability**: Interpret individual predictions.
- **Model Calibration**: LightGBM is calibrated to improve probability outputs.
- **Precision-Recall Optimization**: Custom threshold optimization for improved precision-recall tradeoff.
- **Dockerized**: Containerization for easy replication and future deployment.

## Project Setup

To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Issac-Kondreddy/fraud-detection-system.git
    cd fraud-detection-system
    ```

2. **Install Dependencies**:
    - Install Python packages:
      ```bash
      pip install -r requirements.txt
      ```

3. **Run Django Migrations**:
    ```bash
    python manage.py migrate
    ```

4. **Run the Django Server**:
    ```bash
    python manage.py runserver
    ```

---

## Usage

After the server is running, you can make predictions by sending a POST request to the `/predict/` endpoint.

### Example Request:

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
-H 'Content-Type: application/json' \
-d '{
  "feature1": value1,
  "feature2": value2,
  ...
}'
Example Response:
```bash
{
  "prediction": "Fraud" or "Not Fraud",
  "probability": 0.85
}
```
## Model Information
LightGBM Model: This model uses 400 estimators, a max depth of 15, and balanced class weights to handle the imbalanced nature of fraud detection data. It is calibrated using sigmoid calibration.
Neural Network: A simple 3-layer neural network with Dropout and ReLU activation, trained on a hybrid dataset (original features + LightGBM output).
LIME Explainability: LIME provides feature importance on a per-instance basis for enhanced transparency in model decisions.
Model Performance
The hybrid model (LightGBM + Neural Network) has the following performance metrics on the test data:

**Accuracy:** 1.000
**Precision:** 0.999
**Recall:** 1.000
**F1-Score:** 1.000
Optimal Threshold: 0.85 (calculated from precision-recall curve).

## Confusion Matrix:
|                    | Predicted Not Fraud | Predicted Fraud |
|--------------------|---------------------|-----------------|
| **Actual Not Fraud** | 62675               | 22              |
| **Actual Fraud**     | 4                   | 34123           |

---

## ROC Curve:
The model achieves a high **AUC-ROC** score, demonstrating its ability to discriminate between fraudulent and non-fraudulent transactions.

---

## Precision-Recall Curve:
Optimal precision-recall tradeoff was identified using a threshold of **0.85**.

---

## API Endpoints
- **`/predict/`**: POST request to predict whether a transaction is fraudulent.
    - **Input**: JSON with the required features.
    - **Output**: JSON with the prediction (`"Fraud"` or `"Not Fraud"`) and the confidence score.

---

## Docker Instructions

To run the project using Docker:

1. **Build the Docker image**:
    ```bash
    docker build -t fraud-detection-system .
    ```

2. **Run the Docker container**:
    ```bash
    docker run -p 8000:8000 fraud-detection-system
    ```

The API will now be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

If using `docker-compose`, simply run:
```bash
docker-compose up
```
## Technologies Used
Python 3.x
LightGBM: Gradient boosting model for fraud detection.
Neural Network (Keras): A fully connected neural network for hybrid model prediction.
LIME: Used for explainability of individual predictions.
Django: Backend web framework to serve the model.
Docker: For containerizing the project.

## Future Improvements
1. Deploy to Cloud: Future deployment to AWS, GCP, or Azure.
2. Real-Time Predictions: Add streaming support for real-time fraud detection.
3. Advanced Tuning: Further hyperparameter tuning or exploring advanced models (e.g., XGBoost).
