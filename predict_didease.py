import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from mlxtend.plotting import plot_sequential_feature_selection
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
df = pd.read_csv('heart_attack_prediction_dataset.csv')

print(df.columns)
df['Diet'] = df['Diet'].replace({'Healthy':1,'Average':2,'Unhealthy':3})
df['Continent'] = df['Continent'].replace({'South America':1,'North America':2,'Europe':3,'Asia':4,'Africa':5,'Australia':6})
df['Sex'] = df['Sex'].replace({'Male':1,'Female':0})
df_encoded = df.drop(columns=['Hemisphere','Country','Blood Pressure', 'Patient ID'])
correlation_matrix = df_encoded.corr()

print(df.info())
red_blue = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(correlation_matrix, vmin = -1, vmax = 1, cmap=red_blue)
#plt.show()
# Define features and target variable
X = df[['Age', 'Sex', 'Heart Rate']] # Adjust features as needed
#X = df[['Sex', 'Alcohol Consumption', 'Medication Use']]
y = df['Heart Attack Risk']  # Assuming 'Sex' is the binary target variable

# Apply feature scaling to all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Logistic Regression model
lin_reg = LogisticRegression(class_weight='balanced')

# Initialize Sequential Feature Selector
sfs = SFS(lin_reg, k_features=3, forward=True, floating=False, cv=5, scoring='accuracy')

# Fit Sequential Feature Selector
sfs.fit(X_scaled, y)

# Get selected feature indices
selected_features = list(sfs.k_feature_idx_)
selected_features = [X.columns[i] for i in selected_features]
print(selected_features)
print(sfs.subsets_)
plot_sequential_feature_selection(sfs.get_metric_dict())
plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=6)

# Fit the Logistic Regression model on selected features
lin_reg.fit(X_train, y_train)

# Make predictions
y_pred = lin_reg.predict(X_test)
y_prob = lin_reg.predict_proba(X_test)[:, 1]
threshold = 0.5082

y_pred = (y_prob > threshold).astype(int)
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
print(df['Heart Attack Risk'].value_counts())
C_array  = np.logspace(-4, -2, 100)
#Making a dict to enter as an input to param_grid
tuning_C = {'C':C_array}
from sklearn.model_selection import GridSearchCV
clf_gs = LogisticRegression()
gs = GridSearchCV(clf_gs, param_grid=tuning_C, scoring ='f1', cv = 5)
gs.fit(X_train,y_train)
## 12. Optimal C value and the score corresponding to it
print(gs.best_params_, gs.best_score_)

clf_best = LogisticRegression(C=gs.best_params_['C'])
clf_best.fit(X_train,y_train)
y_pred_best = clf_best.predict(X_test)
print(f1_score(y_test,y_pred_best))
#print(X.head())
#print(y.head())

#clf_best.predict(gg)
#print(clf_best.predict(gg))
