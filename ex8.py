import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the iris dataset
dataset = load_iris()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Create and train the KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

# Streamlit app
st.title("Iris Dataset KNN Classifier")
st.write("This app uses a K-Nearest Neighbors classifier to predict the species of iris flowers.")

# Inputs for the user to enter new iris flower measurements
st.sidebar.header("Enter Iris Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(dataset["data"][:, 0].min()), float(dataset["data"][:, 0].max()), float(dataset["data"][:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(dataset["data"][:, 1].min()), float(dataset["data"][:, 1].max()), float(dataset["data"][:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(dataset["data"][:, 2].min()), float(dataset["data"][:, 2].max()), float(dataset["data"][:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(dataset["data"][:, 3].min()), float(dataset["data"][:, 3].max()), float(dataset["data"][:, 3].mean()))

# Collect user input into a numpy array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict the class of the input data
prediction = kn.predict(input_data)
predicted_species = dataset["target_names"][prediction][0]

st.write(f"The predicted species is: **{predicted_species}**")

# Display the accuracy of the model
accuracy = kn.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Display a dataframe with some of the test set predictions as an example
st.write("Example predictions from the test set:")

# Create a dataframe to store the results
results = []
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    results.append({
        "Sepal Length": x[0],
        "Sepal Width": x[1],
        "Petal Length": x[2],
        "Petal Width": x[3],
        "Actual Species": dataset["target_names"][y_test[i]],
        "Predicted Species": dataset["target_names"][prediction][0]
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df.head(10))  # Show the first 10 predictions

