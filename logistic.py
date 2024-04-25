import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
df = pd.read_csv("Eartquakes-1990-2023.csv")
scaler = StandardScaler()
X = df[['time', 'significance', 'longitude', 'latitude', 'depth']]
y = df['tsunami']
##sfs = SFS(logi, k_features=5, # number of features to select
#            forward=True , # ensure that we are using sequential forward selection
#            floating=False, # ensure that we are using sequential forward selection not floating
#            scoring='accuracy', # determines how the algorithm will evaluate each feature subset. It’s often okay to use the default value None because mlxtend will automatically use a metric that is suitable for whatever scikit-learn model you are using. For this lesson, we’ll set it to 'accuracy'.
#            cv=0)
## Fit the equential forward selection model
#sfs.fit(X_scaled,y)
## Print the chosen feature names
#print(sfs.subsets_)
#selected_features = list(sfs.k_feature_idx_)
#selected_features = [X.columns[i] for i in selected_features]
#print(selected_features)
#plot_sequential_feature_selection(sfs.get_metric_dict())
#plt.show()
# C_array  = np.logspace(-4, -2, 100)
# #Making a dict to enter as an input to param_grid
# tuning_C = {'C':C_array}
# from sklearn.model_selection import GridSearchCV
# clf_gs = LogisticRegression()
# gs = GridSearchCV(clf_gs, param_grid=tuning_C, scoring ='f1', cv = 6)
# gs.fit(X_train,y_train)
# ## 12. Optimal C value and the score corresponding to it
# print(gs.best_params_, gs.best_score_)
#
# clf_best = LogisticRegression(C=gs.best_params_['C'])
# clf_best.fit(X_train,y_train)
# y_pred_best = clf_best.predict(X_test)
# print(f1_score(y_test,y_pred_best))
X_scaled = scaler.fit_transform(X)
logi = LogisticRegression(class_weight='balanced', C=0.008697490026177835, max_iter=1000)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=6)
logi.fit(X_train,y_train)
y_pred = logi.predict(X_test)
#y_prob = logi.predict_proba(X_test)[:, 1]
threshold = 0.958
#y_pred = (y_prob > threshold).astype(int)
# Evaluate the model
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
kk = [[1408565558160, 465, 175.378, 52.5074, 5.19], [1338508907690, 68, -122.809, 38.8363, 169]]
new_data_scaled = scaler.transform(kk)  # Apply the same scaling as your training data
new_data_predict = logi.predict_proba(new_data_scaled)

# Define your new threshold
new_threshold = 0.958  # Adjust this value as needed

# Apply the threshold to classify the new data
new_predictions = (new_data_predict[:, 1] > new_threshold).astype(int)

print("Predicted probabilities for new data:")
print(new_data_predict)
print("Predicted classes for new data with the new threshold:")
print(new_predictions)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)
poly_reg = LogisticRegression()
poly_reg.fit(X_train_poly, y_train)
y_prob = poly_reg.predict_proba(X_test_poly)[:, 1]
threshold = 0.5112


y_pred = poly_reg.predict(X_test_poly)
y_pred = (y_prob > threshold).astype(int)
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
kk = [[1408565558160, 465, 175.378, 52.5074, 5.19],[1338508907690, 68, -122.809, 38.8363, 169]]
kk_scaler = StandardScaler()

# Fit the kk_scaler with your training data
kk_scaler.fit(X_train)

# Scale the kk data
kk_scaled = kk_scaler.transform(kk)


# Continue with transformation and prediction as before
kk_poly = poly.transform(kk_scaled)
predicted = poly_reg.predict(kk_poly)
print(predicted)