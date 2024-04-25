import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df = pd.read_csv("winequality-red.csv")
print(df.describe())
print(df.columns)
correlation_matrix = df.corr()
sns.boxplot(x=df["fixed acidity"])
plt.show()
df_corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()
sns.scatterplot(x=df["fixed acidity"], y=df['citric acid'])
plt.show()
print(df["fixed acidity"].corr(df['citric acid']))
print(df["fixed acidity"].corr(df['density']))
print(df["free sulfur dioxide"].corr(df['total sulfur dioxide']))
X = df.drop(columns=["quality"])
y = df["quality"]
print(X.shape)
print(y.shape)
print(df.isnull().sum())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree1 = DecisionTreeClassifier(criterion="gini", max_depth=7)
dtree1.fit(X_train, y_train)
dtree1_depth = dtree1.get_depth()
print(f'First Decision Tree depth: {dtree1_depth}')
dtree1_score = dtree1.score(X_test, y_test)
print(f'Test set accuracy tree no max depth: {dtree1_score}')
