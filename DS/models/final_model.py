import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib

# Load preprocessed data
preprocessed_data = pd.read_csv(r'/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/data/processed_creditcard.csv')

# Separate features and target
X = preprocessed_data.drop('Class', axis=1)
y = preprocessed_data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train LightGBM (GBDT) model
lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
lgb_model.fit(X_train, y_train)

# Get GBDT predictions (probabilities)
train_gbdt_predictions = lgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
test_gbdt_predictions = lgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

# Create hybrid datasets (GBDT output + original features)
X_train_hybrid = np.hstack([X_train, train_gbdt_predictions])
X_test_hybrid = np.hstack([X_test, test_gbdt_predictions])

# Define the Neural Network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_hybrid.shape[1], activation='relu'))  # Input size matches the hybrid feature set
model.add(Dropout(0.3))  # Regularization
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile the NN model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the NN model
history = model.fit(X_train_hybrid, y_train, epochs=10, batch_size=32, validation_data=(X_test_hybrid, y_test))

# Predict probabilities and labels
y_pred_proba = model.predict(X_test_hybrid).ravel()  # Get predicted probabilities
y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert to binary (0 or 1)

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
plt.figure(figsize=(8,6))
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
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label='GBDT + NN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the trained models
joblib.dump(lgb_model, 'gbdt_model.pkl')  # Save the LightGBM model
model.save('nn_model.keras')  # Save the Neural Network model

# Print performance metrics
report = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy: {report['accuracy']:.3f}")
print(f"Precision: {report['1']['precision']:.3f}")
print(f"Recall: {report['1']['recall']:.3f}")
print(f"F1-Score: {report['1']['f1-score']:.3f}")
