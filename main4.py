# SVM algorithm with Grid Search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Loading the dataset
data = pd.read_csv("emails.csv")

# Dropping email number
data = data.drop(columns=["Email No."])

# Separating dependent and independent variables
X = data.drop(columns=["Prediction"])
y = data["Prediction"]

# 90% dataset for training and 10% for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Defining the parameter grid with a smaller search space
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Printing the best parameters and best score
print(f"Best parameters from Grid Search: {grid_search.best_params_}")
print(f"Best score from Grid Search: {grid_search.best_score_}")

# Using the best estimator to make predictions
best_model = grid_search.best_estimator_
predicted = best_model.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
