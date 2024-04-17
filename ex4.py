import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title('Naive Bayes Classifier')

# Show dataset information
st.write('Iris dataset information:')
st.write(iris.DESCR)

# Show accuracy
st.write('Accuracy:', accuracy)

# Allow user to input new data for prediction
st.sidebar.title('Input New Data')
sepal_length = st.sidebar.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Make prediction on new data
new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = classifier.predict(new_data)

# Show prediction
st.sidebar.write('Predicted Iris Species:', iris.target_names[prediction[0]])
