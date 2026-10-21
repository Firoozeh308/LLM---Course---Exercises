# Exercise 01: Classification of Iris Flowers using K-Nearest Neighbors (KNN)
# Course: Deep Learning with TensorFlow & Keras (Instructor: Alireza Akhavan)

import numpy as np
import pandas as pd
import mglearn 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load the Iris dataset
iris_dataset = load_iris() 
print("Keys of iris_dataset:\n", iris_dataset.keys())

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 3. Create a dataframe from the training data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# 4. Create a scatter matrix from the dataframe, color by y_train
# Note: This requires matplotlib to be installed to display.
pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, figsize=(15, 15), marker='o', 
    hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3
)
# plt.show() # Uncomment this line if running locally

# 5. Train the KNN model
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train, y_train) 

# 6. Make predictions and evaluate the model
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

# 7. Calculate the accuracy score
score = knn.score(X_test, y_test)
print("Test set score: {:.2f}".format(score))