import streamlit as st
import pandas as pd
import math

class Node:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        self.feature = feature          # Feature index
        self.value = value              # Threshold value if the feature is numeric
        self.results = results          # Results for leaf node (class probabilities)
        self.true_branch = true_branch  # True branch (greater or equal)
        self.false_branch = false_branch # False branch (less than)

def calculate_entropy(data):
    # Calculate entropy of a dataset
    total_count = len(data)
    label_counts = data['label'].value_counts()
    entropy = 0
    for label in label_counts:
        p = label / total_count
        entropy -= p * math.log2(p)
    return entropy

def calculate_information_gain(data, feature, value):
    # Calculate information gain for a specific feature and value
    true_data = data[data[feature] >= value]
    false_data = data[data[feature] < value]
    
    # Calculate entropy for the true and false branches
    entropy_true = calculate_entropy(true_data)
    entropy_false = calculate_entropy(false_data)
    
    # Calculate total entropy after the split
    p_true = len(true_data) / len(data)
    p_false = len(false_data) / len(data)
    total_entropy = p_true * entropy_true + p_false * entropy_false
    
    # Calculate information gain
    parent_entropy = calculate_entropy(data)
    information_gain = parent_entropy - total_entropy
    
    return information_gain

def get_best_split(data, features):
    # Find the best split for the dataset
    best_gain = 0
    best_feature = None
    best_value = None
    
    for feature in features:
        values = data[feature].unique()
        for value in values:
            gain = calculate_information_gain(data, feature, value)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value
    
    return best_feature, best_value

def build_tree(data, features):
    # Recursive function to build the decision tree
    if len(data['label'].unique()) == 1: # If only one class in the data, return a leaf node
        return Node(results=data['label'].value_counts(normalize=True).to_dict())
    
    if len(features) == 0: # If no features left to split on, return a leaf node with majority class
        return Node(results=data['label'].value_counts(normalize=True).to_dict())
    
    best_feature, best_value = get_best_split(data, features)
    if best_feature is None: # If unable to find a split, return a leaf node with majority class
        return Node(results=data['label'].value_counts(normalize=True).to_dict())
    
    true_data = data[data[best_feature] >= best_value]
    false_data = data[data[best_feature] < best_value]
    
    true_branch = build_tree(true_data, features)
    false_branch = build_tree(false_data, features)
    
    return Node(feature=best_feature, value=best_value, true_branch=true_branch, false_branch=false_branch)

def classify(tree, sample):
    # Classify a sample using the decision tree
    if tree.results is not None:
        return max(tree.results, key=tree.results.get)
    
    if sample[tree.feature] >= tree.value:
        return classify(tree.true_branch, sample)
    else:
        return classify(tree.false_branch, sample)

def main():
    st.title("ID3 Decision Tree with Streamlit")

    # Example dataset
    data = pd.DataFrame({
        'feature1': [1, 1, 0, 0, 1, 1, 0, 0],
        'feature2': [1, 1, 1, 0, 0, 1, 1, 0],
        'label': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B']
    })

    st.write("Preview of the dataset:")
    st.dataframe(data)

    # Build the decision tree
    tree = build_tree(data, ['feature1', 'feature2'])

    # Classify a sample
    sample = {'feature1': 0, 'feature2': 1}
    prediction = classify(tree, sample)
    st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
