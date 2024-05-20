import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read the CSV file
@st.cache
def load_data():
    data = pd.read_csv("heartdisease.csv")
    return pd.DataFrame(data)

heart_disease = load_data()

# Define Bayesian Network
model = BayesianModel([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')
])

# Fit model
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Inference Engine
HeartDisease_infer = VariableElimination(model)

# Display inputs and get user inputs
st.write('Enter the following details:')
age = st.selectbox('Age', ['SuperSeniorCitizen', 'SeniorCitizen', 'MiddleAged', 'Youth', 'Teen'])
gender = st.selectbox('Gender', ['Male', 'Female'])
family = st.selectbox('Family History', ['No', 'Yes'])
diet = st.selectbox('Diet', ['High', 'Medium'])
lifestyle = st.selectbox('Lifestyle', ['Athlete', 'Active', 'Moderate', 'Sedentary'])
cholesterol = st.selectbox('Cholesterol', ['High', 'BorderLine', 'Normal'])

# Map user inputs to numerical values
age_map = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
gender_map = {'Male': 0, 'Female': 1}
family_map = {'No': 0, 'Yes': 1}
diet_map = {'High': 0, 'Medium': 1}
lifestyle_map = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedentary': 3}
cholesterol_map = {'High': 0, '
