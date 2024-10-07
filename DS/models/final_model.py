import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib
import lime
from lime.lime_tabular import LimeTabularExplainer
# Load preprocessed data
preprocessed_data = pd.read_csv(r'/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/data/processed_creditcard.csv')

# Separate features and target
X = preprocessed_data.drop('Class', axis=1)
y = preprocessed_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert y_train and y_test to numpy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Train LightGBM (GBDT) model
lgb_model = lgb.LGBMClassifier(n_estimators=400, max_depth=15, learning_rate=0.02, class_weight='balanced')
lgb_model.fit(X_train, y_train)

# Apply calibration on the GBDT model
calibrated_gbdt = CalibratedClassifierCV(lgb_model, method='sigmoid', cv=5)
calibrated_gbdt.fit(X_train, y_train)

# Get calibrated GBDT predictions (probabilities)
train_gbdt_predictions = calibrated_gbdt.predict_proba(X_train)[:, 1].reshape(-1, 1)
test_gbdt_predictions = calibrated_gbdt.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Create hybrid datasets (GBDT output + original features)
X_train_hybrid = np.hstack([X_train, train_gbdt_predictions])
X_test_hybrid = np.hstack([X_test, test_gbdt_predictions])

# Define the Neural Network model
model = Sequential([
    Dense(64, input_shape=(X_train_hybrid.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the NN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_nn_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

# Train the NN model with callbacks to prevent overfitting
history = model.fit(
    X_train_hybrid, y_train,  # Using converted numpy arrays
    epochs=20,
    batch_size=32,
    validation_data=(X_test_hybrid, y_test),
    callbacks=[early_stopping, checkpoint],
    class_weight={0: 10, 1: 25}  # Increase weight for fraud class to ensure balanced learning
)

# Predict probabilities and labels
y_pred_proba = model.predict(X_test_hybrid).ravel()  # Get predicted probabilities

#
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)  # Avoid division by zero
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.2f}")

# Update predictions based on the optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate AUC-ROC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Generate ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'GBDT + NN (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random model baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot accuracy and loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='GBDT + NN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the trained models
joblib.dump(calibrated_gbdt, 'calibrated_gbdt_model.pkl')  # Save the calibrated LightGBM model
model.save('nn_model.keras')  # Save the Neural Network model in .keras format

# Print performance metrics
report = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy: {report['accuracy']:.3f}")
print(f"Precision: {report['1']['precision']:.3f}")
print(f"Recall: {report['1']['recall']:.3f}")
print(f"F1-Score: {report['1']['f1-score']:.3f}")

# Create a LIME explainer for tabular data
explainer = LimeTabularExplainer(
    training_data=X_train_hybrid,  # The training data
    training_labels=y_train,       # The corresponding labels
    mode='classification',         # This is a classification task
    feature_names=preprocessed_data.columns.tolist() + ['GBDT_output'],  # Feature names + GBDT output
    class_names=['Not Fraud', 'Fraud'],  # Classes for fraud detection
    discretize_continuous=True
)

# Choose a sample from your test set for explanation
sample_index = 0  # You can select any index from the test set to explain
sample_data = X_test_hybrid[sample_index].reshape(1, -1)

# Define a prediction function that returns probabilities using model.predict()
def predict_fn(data):
    # Get the probability of class 1 (fraud) using model.predict()
    class_1_proba = model.predict(data).reshape(-1, 1)
    
    # Class 0 probability is just 1 - class 1 probability
    class_0_proba = 1 - class_1_proba
    
    # Return both probabilities as a 2D array
    return np.hstack([class_0_proba, class_1_proba])


# Use LIME to explain the prediction for this instance
explanation = explainer.explain_instance(
    data_row=sample_data[0],  # The sample data you want to explain
    predict_fn=predict_fn,  # Use updated prediction function
    num_features=10  # Show top 10 most important features
)

# Show the explanation
explanation.show_in_notebook(show_table=True, show_all=False)

# Save the explanation to a file if not using a notebook
explanation.save_to_file('lime_explanation.html')