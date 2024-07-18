# SVM algorithm with Random Search
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import expon

# Loading the dataset
data = pd.read_csv("emails.csv")

# Dropping email number
data = data.drop(columns=["Email No."])

# Separating dependent and independent variables
X = data.drop(columns=["Prediction"])
y = data["Prediction"]

# 90% dataset for training and 10% for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the parameter distribution
param_dist = {
    'C': expon(scale=100),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]  # Only relevant for 'poly' kernel
}

# Initialize the RandomizedSearchCV object
random_search = RandomizedSearchCV(SVC(random_state=42), param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42)

# Fit the random search to the data
random_search.fit(x_train, y_train)

# Printing the best parameters and best score
print(f"Best parameters from Random Search: {random_search.best_params_}")
print(f"Best score from Random Search: {random_search.best_score_}")

# Use the best estimator to make predictions
best_model = random_search.best_estimator_
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
