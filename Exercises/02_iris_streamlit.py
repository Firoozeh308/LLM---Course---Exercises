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

# --- Custom CSS for Slider Fix ---
st.markdown("""
<style>
    /* Fix slider thumb alignment on mobile */
    div[data-testid="stSlider"] div[role="slider"] {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        align-self: center !important;
    }
    
    /* Make the track (line) and thumb (circle) align properly */
    div[data-testid="stSlider"] > div {
        display: flex !important;
        align-items: center !important;
    }
    
    /* Make slider labels cleaner */
    div[data-testid="stSlider"] p {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #555 !important;
        margin-bottom: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

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