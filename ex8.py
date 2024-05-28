import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Manually defined Iris dataset (sample)
data = {
    'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
                     5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
                     6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0,
                     5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9],
    'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,
                    3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8,
                    3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0,
                    3.0, 2.7, 2.8, 2.2, 3.1, 2.5, 2.7, 2.2, 2.5, 3.2],
    'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,
                     1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5,
                     4.5, 4.9, 4.0, 4.6, 3.5, 4.4, 3.3, 4.6, 3.9, 3.5,
                     4.2, 4.0, 4.7, 3.6, 4.4, 3.9, 4.4, 4.5, 3.9, 4.8],
    'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,
                    0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3,
                    1.5, 1.5, 1.3, 1.5, 1.0, 1.3, 1.0, 1.3, 1.4, 1.0,
                    1.5, 1.0, 1.4, 1.3, 1.4, 1.1, 1.5, 1.5, 1.1, 1.8],
    'species': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

target_names = ["setosa", "versicolor"]

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df.drop(columns=['species'])
y = df['species']

# Train the KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X, y)

# Streamlit app
st.title("Iris Dataset KNN Classifier")
st.write("This app uses a K-Nearest Neighbors classifier to predict the species of iris flowers based on your input measurements.")

# Inputs for the user to enter new iris flower measurements
st.sidebar.header("Enter Iris Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal_length'].min()), float(X['sepal_length'].max()), float(X['sepal_length'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal_width'].min()), float(X['sepal_width'].max()), float(X['sepal_width'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal_length'].min()), float(X['petal_length'].max()), float(X['petal_length'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal_width'].min()), float(X['petal_width'].max()), float(X['petal_width'].mean()))

# Collect user input into a numpy array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict the class of the input data
prediction = kn.predict(input_data)
predicted_species = target_names[prediction[0]]

st.write(f"The predicted species is: **{predicted_species}**")

# Display the accuracy of the model
st.write("Note: Model accuracy is not available because we're using a manually defined subset of data.")

# Optional: Display some of the manually defined data as an example
st.write("Example data:")
st.dataframe(df.head(10))
