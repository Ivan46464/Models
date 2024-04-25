# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
df = pd.read_csv("Eartquakes-1990-2023.csv")

# Define the features and the target
X = df[['time', 'significance', 'longitude', 'latitude', 'depth']]
y = df['tsunami']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_scaled)

# Split the data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X_poly,y,test_size=0.2,random_state=6)

# Create and fit a logistic regression model with balanced class weights and regularization parameter C=0.01
logi = LogisticRegression(class_weight='balanced', C=0.01, max_iter=1000)
logi.fit(X_train,y_train)

# Predict the classes and probabilities on the test set
y_pred = logi.predict(X_test)
y_prob = logi.predict_proba(X_test)[:, 1]

# Define a threshold of 0.958 for classification
threshold = 0.958

# Apply the threshold to the probabilities to get the predictions
y_pred = (y_prob > threshold).astype(int)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix on test set:')
print(confusion_matrix(y_test, y_pred))

# Plot the ROC curve and calculate the AUC score
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Predict on new data kk using the same scaling and polynomial features
kk = [[1408565558160, 465, 175.378, 52.5074, 5.19], [1338508907690, 68, -122.809, 38.8363, 169]]
new_data_scaled = scaler.transform(kk)  # Apply the same scaling as your training data
new_data_poly = poly.transform(new_data_scaled)  # Apply the same polynomial features as your training data
new_data_predict = logi.predict_proba(new_data_poly)

# Define your new threshold
new_threshold = 0.958  # Adjust this value as needed

# Apply the threshold to classify the new data
new_predictions = (new_data_predict[:, 1] > new_threshold).astype(int)

print("Predicted probabilities for new data:")
print(new_data_predict)
print("Predicted classes for new data with the new threshold:")
print(new_predictions)
