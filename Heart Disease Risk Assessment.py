# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the preprocessed heart disease dataset with all 13 features
@st.cache_data
def load_data():
    data = pd.read_csv('heart_disease_dataset.csv')
    return data

heart_data = load_data()

# Split the data into features and target variable
X = heart_data.drop('target', axis=1)  # Features
y = heart_data['target']  # Target variable

# Build the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)  # Train the model on the entire dataset

# Streamlit UI
st.title('Heart Disease Risk Assessment')

# User input for health metrics
st.sidebar.header('Enter Health Metrics')
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120)
chol = st.sidebar.number_input('Serum Cholesterol (mg/dL)', min_value=0, max_value=600, value=200)
fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
exang = st.sidebar.radio('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
thal = st.sidebar.selectbox('Thal', [0, 1, 2, 3])

# Convert user input to numeric values
sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'Yes' else 0
exang = 1 if exang == 'Yes' else 0

# Predict button
if st.sidebar.button('Predict'):
    # Make prediction based on user input
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]

    # Display prediction result
    st.write('## Prediction Result')
    if prediction == 0:
        st.write('No heart disease risk detected.')
    else:
        st.write('Heart disease risk detected.')
