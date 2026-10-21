import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

@st.cache_resource
def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    return knn, iris.target_names

knn, target_names = train_model()

st.title("🌸 Iris Flower Classification")
st.write("Adjust the sliders to predict the species of the Iris flower.")

# Using number inputs for a cleaner mobile experience
st.markdown("### Enter Flower Measurements")
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(features)[0]
    st.success(f"The predicted species is: **{target_names[prediction].capitalize()}**")