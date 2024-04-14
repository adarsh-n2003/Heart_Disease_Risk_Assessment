### **Heart Disease Risk Assessment Project**

This repository contains code for a heart disease risk assessment tool, enabling users to input health metrics through a user-friendly interface and utilizing a machine learning model (Logistic Regression) for risk assessment.

### **Dataset**

The dataset used in this project is obtained from Kaggle and contains anonymized patient data relevant to the diagnosis of heart disease. It includes features such as age, sex, blood pressure, cholesterol levels, and more. The dataset is sourced from the UCI Machine Learning Repository.

You can access the dataset on Kaggle via the following link: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### **Project Overview**

The project involves the following steps:

1. **User Interface Development:** Developing a user-friendly interface using Streamlit to allow users to conveniently input their health metrics.
2. **Model Building:** Creating a machine learning model using the Logistic Regression algorithm to predict the risk of heart disease based on the input health metrics.
3. **Model Deployment:** Deploying the model within the Streamlit application to provide real-time risk assessment.

### **Files**

- **`heart_disease_risk_assessment.py`**: Python script containing the code for the user interface, model building, and deployment.
- **`heart_disease_dataset.csv`**: CSV file containing the heart disease patient data.

### **Usage**

To use the code in this repository:

1. Clone the repository to your local machine.
2. Download the **`heart_disease_dataset.csv`** file from the provided Kaggle link and place it in the repository directory.
3. Install the required dependencies by running the following command:
    
    ```
    Copy code
    pip install streamlit pandas numpy scikit-learn
    ```
    
4. Run the **`heart_disease_risk_assessment.py`** script to launch the application.

### **Requirements**

The project code requires the following Python libraries:

- streamlit
- pandas
- numpy
- scikit-learn

You can install these dependencies using pip:

```
Copy code
pip install streamlit pandas numpy scikit-learn
```

### **Contributor**

- [Adarsh Nashine](https://github.com/adarsh-n2003)

### **Acknowledgments**

- Thanks to Kaggle and the UCI Machine Learning Repository for providing the dataset used in this project.
- Special thanks to the contributors and maintainers of the dataset for making it publicly available for research and educational purposes.

### **License**

This project is licensed under the MIT License. Feel free to use the code for educational and commercial purposes.
