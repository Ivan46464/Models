import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mlxtend.plotting import plot_sequential_feature_selection

df = pd.read_csv("heart_attack_prediction_dataset.csv")
print(df.head())
print(df.info())
df1 = df.drop(columns=["Hemisphere", 'Continent', 'Country', 'Diet', 'Blood Pressure', 'Sex', 'Patient ID'])
std_df = []
for i in df1.columns:
    std_any = df1[i].describe()
    std_df.append(std_any['std'])
labels = df1.columns
print(std_df)

plt.bar(labels, std_df)
plt.xlabel("Columns")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=90)
plt.show()
plt.clf()
sns.countplot(x=df["Continent"], hue=df["Sex"])
plt.show()
plt.clf()
sns.histplot(x=df["Income"],bins=10)
plt.show()
print(df["Income"].head())
df["Income"] = np.log(df["Income"])
print(df["Income"].std())
print(df["Income"].head())

sns.histplot(x=df["Income"],bins=10)
plt.show()
print(df["Continent"].value_counts())
df['Continent'] = df['Continent'].replace({'South America':1,'North America':2,'Europe':3,'Asia':4,'Africa':5,'Australia':6})
print(df["Diet"].value_counts())
df["Diet"] = df["Diet"].replace({"Healthy":1, "Average":2, "Unhealthy":3})
print(df["Sex"].value_counts())
df["Sex"] = df["Sex"].replace({"Male":1, "Female":0})
print(df["Blood Pressure"].head())
df["Blood Pressure"] = df["Blood Pressure"].str.split("/")
print(df["Blood Pressure"].head())
df["Systolic"] = df["Blood Pressure"].apply(lambda x: float(x[0]))
df["Diastolic"] = df["Blood Pressure"].apply(lambda x: float(x[1]))

# Calculate the ratio of systolic and diastolic values and store it in a new column
df["Ratio"] = df["Systolic"] / df["Diastolic"]
print(df["Ratio"].head())
print(df["Hemisphere"].value_counts())
df["Hemisphere"] = df["Hemisphere"].replace({"Northern Hemisphere": 1, "Southern Hemisphere": 0})
print(df["Country"].value_counts())
print(df["Country"].nunique())
print(df.columns)
df_for_X = df.drop(columns=["Blood Pressure",'Systolic', 'Diastolic', "Country", 'Patient ID', "Heart Attack Risk"])
X = df_for_X
y = df["Heart Attack Risk"]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
# avgScore = []
# k_list = [i for i in range(1,101)]
# for i in range(1,101):
#     KNN = KNeighborsClassifier(n_neighbors=i)
#     from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#     sfs(KNN, k_features=10, forward=True, floating=False, cv=5, scoring='accuracy')
#     sfs.fit(X_scaled, y)
#     subsets = sfs.subsets_
#     avg_score = sum(subset['avg_score'] for subset in subsets.values()) / len(subsets)
#     avgScore.append(avg_score)
#
# plt.plot(k_list, avgScore)
# plt.xlabel("k")
# plt.ylabel("Validation Accuracy")
# plt.title("Breast Cancer Classifier Accuracy")
# plt.show()
# print(max(avgScore))

# KNN = KNeighborsClassifier(n_neighbors=74)
# KNN.fit(X, y)
# y_test_pred = KNN.predict(X_test)
# from sklearn.metrics import accuracy_score
#
# # Calculate and print the accuracy score
# accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Accuracy: {accuracy}")
#
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#
# print(confusion_matrix(y_test, y_test_pred))
KNN = KNeighborsClassifier(n_neighbors=74)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(KNN, k_features=10, forward=True, floating=False, cv=5, scoring='accuracy')

sfs.fit(X_scaled, y)

selected_features = list(sfs.k_feature_idx_)
selected_features = [X.columns[i] for i in selected_features]
print(selected_features)
print(sfs.subsets_)

New_X = df[['Sex', 'Heart Rate', 'Smoking', 'Obesity']]
New_y = df["Heart Attack Risk"]
New_X_scaled = scaler.fit_transform(X)
New_X_train, New_X_test, New_y_train, New_y_test = train_test_split(New_X_scaled,New_y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
KNN1 = KNeighborsClassifier(n_neighbors=74)
KNN1.fit(X, y)
y_test_pred = KNN1.predict(X_test)
from sklearn.metrics import accuracy_score
# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")