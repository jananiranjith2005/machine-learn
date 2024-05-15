import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

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

    # Show accuracy
    st.write('Accuracy:', accuracy)

    # Allow user to input new data for prediction
    st.sidebar.title('Input New Data')
    feature1 = st.sidebar.slider('Feature 1', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    feature2 = st.sidebar.slider('Feature 2', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))

    # Make prediction on new data
    new_data = np.array([[feature1, feature2]])
    prediction = classifier.predict(new_data)

    # Show prediction
    st.sidebar.write('Predicted Class:', prediction[0])

if __name__ == "__main__":
    main()
