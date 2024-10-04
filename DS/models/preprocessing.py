import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv(r'/Users/issackondreddy/Desktop/Projects/Fraud Detection System/DS/data/creditcard.csv')
print("Original Dataset shape:", data.shape)

# Initialize the scaler
scaler = StandardScaler()

# Scale the 'Amount' and 'Time' columns and add to the dataset
data['Scaled_Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Scaled_Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original 'Amount' and 'Time' columns
data = data.drop(['Amount', 'Time'], axis=1)

# Reorganize columns: Scaled_Time, Scaled_Amount first, then V1, V2, ..., V28, and finally Class
scaled_columns = ['Scaled_Time', 'Scaled_Amount'] + [col for col in data.columns if col not in ['Scaled_Time', 'Scaled_Amount', 'Class']]
X = data[scaled_columns]
y = data['Class']

# Fit the scaler on the entire feature set (after 'Amount' and 'Time' are scaled)
scaler.fit(X)  # Now fit on the entire set of features

# Save the fitted scaler for later use in predictions
joblib.dump(scaler, 'DS/models/scaler.pkl')  # Save the scaler

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert resampled data back to DataFrame
data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
data_resampled['Class'] = y_resampled

# Detecting outliers using IQR (from the resampled data)
Q1 = data_resampled.quantile(0.25)
Q3 = data_resampled.quantile(0.75)
IQR = Q3 - Q1
data_no_outliers = data_resampled[~((data_resampled < (Q1 - 1.5 * IQR)) | (data_resampled > (Q3 + 1.5 * IQR))).any(axis=1)]

print("Dataset shape after removing outliers:", data_no_outliers.shape)

# Save the preprocessed and resampled dataset to a CSV file
data_no_outliers.to_csv('DS/data/processed_creditcard.csv', index=False)
print("Preprocessed data saved to 'DS/data/processed_creditcard.csv'")