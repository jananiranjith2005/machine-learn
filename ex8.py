import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

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

# Display a subset of the test set with predictions
st.write("Test Set Predictions:")

# Create a dataframe to store the results
import pandas as pd

results = []
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    results.append({
        "TARGET": y_test[i],
        "TARGET_NAME": dataset["target_names"][y_test[i]],
        "PREDICTED": prediction[0],
        "PREDICTED_NAME": dataset["target_names"][prediction][0]
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Display the accuracy of the model
accuracy = kn.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")
