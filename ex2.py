import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import math

def calculate_entropy(data, target_column):
    total_rows = len(data)
    target_values = data[target_column].unique()

    entropy = 0
    for value in target_values:
        # Calculate the proportion of instances with the current value
        value_count = len(data[data[target_column] == value])
        proportion = value_count / total_rows
        entropy -= proportion * math.log2(proportion)

    return entropy

def main():
    st.title("Decision Tree Visualization")

    # Load data
    df = pd.read_csv('diabetes.csv')

    st.write("Preview of the dataset:")
    st.dataframe(df.head())

    # Calculate entropy of the dataset
    entropy_outcome = calculate_entropy(df, 'Outcome')
    st.write(f"Entropy of the dataset: {entropy_outcome}")

    # Feature selection for the first step in making decision tree
    selected_feature = 'DiabetesPedigreeFunction'

    # Create a decision tree
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    X = df[[selected_feature]]
    y = df['Outcome']
    clf.fit(X, y)

    # Plot the decision tree
    st.write("Decision Tree Visualization:")
    plt.figure(figsize=(8, 6))
    plot_tree(clf, feature_names=[selected_feature], class_names=['0', '1'], filled=True, rounded=True)
    st.pyplot()

if __name__ == "__main__":
    main()
