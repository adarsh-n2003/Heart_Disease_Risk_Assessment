import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import mysql.connector

# Establish connection to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Adarsh@2003",
    database="heartdiseases"
)

# Function to load data
@st.cache_resource
def load_data():
    data = pd.read_csv('heart_disease_dataset.csv')
    return data

# Function to train the model
@st.cache_data
def get_model(X, y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creating a pipeline for scaling and logistic regression
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    pipeline.fit(X_train, y_train)
    
    return pipeline

# Load and prepare data
heart_data = load_data()
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Train model
model = get_model(X, y)

# Streamlit UI setup
st.title('Heart Disease Risk Assessment')

with st.form("prediction_form"):
    st.header('Enter Your Health Metrics:')
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.radio('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=300, value=120)
    chol = st.number_input('Serum Cholesterol (mg/dL)', min_value=100, max_value=600, value=200)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=250, value=150)
    exang = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    thal = st.selectbox('Thal', [0, 1, 2, 3])

    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)[0]
        
        # Convert categorical data to numerical values
        sex = 1 if sex == 'Male' else 0
        fbs = 1 if fbs == 'Yes' else 0
        exang = 1 if exang == 'Yes' else 0
    
        # Prepare data for insertion
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    
        # Insert data into MySQL database
        cursor = conn.cursor()
        insert_query = "INSERT INTO health_metrics (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_query, input_data[0])
        conn.commit()
        cursor.close()
        st.success("Data successfully stored in the database.")

        if prediction == 0:
            st.success('No heart disease risk detected.')
        elif prediction == 1:
            st.error('Heart disease risk detected.')

        conn.close()
