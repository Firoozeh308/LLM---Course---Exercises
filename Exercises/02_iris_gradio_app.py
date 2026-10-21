import gradio as gr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. Train the model (Same as Exercise 01)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0
)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 2. Define the prediction function for the UI
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create an array with the user inputs
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction_index = knn.predict(features)[0]
    # Return the actual name of the flower (setosa, versicolor, or virginica)
    return iris.target_names[prediction_index]

# 3. Create the Gradio Interface
app = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Slider(4.0, 8.0, label="Sepal Length (cm)", step=0.1),
        gr.Slider(2.0, 4.5, label="Sepal Width (cm)", step=0.1),
        gr.Slider(1.0, 7.0, label="Petal Length (cm)", step=0.1),
        gr.Slider(0.1, 2.5, label="Petal Width (cm)", step=0.1)
    ],
    outputs="text",
    title="Iris Flower Classification",
    description="Adjust the sliders and see what kind of Iris flower the AI predicts!"
)

# Run the app
if __name__ == "__main__":
    app.launch()