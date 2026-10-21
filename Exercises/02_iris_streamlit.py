import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# --- Custom CSS for better UI ---

st.markdown("""
<style>
    /* Reduce spacing between sliders */
    div[data-testid="stSlider"] {
        margin-bottom: -15px;
    }
    /* Make slider text smaller and cleaner */
    div[data-testid="stSlider"] p {
        font-size: 14px;
        font-weight: 600;
        color: #555;
    }
    /* Make the main title look nicer */
    h1 {
        color: #6C63FF;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
# --------------------------------

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

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(features)[0]
    st.success(f"The predicted species is: **{target_names[prediction].capitalize()}**")