import streamlit as st
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Disease Prediction App",
    layout="wide",
    page_icon="🩺"
)

# Load models

model_pneumonia = load_model('pneumonia_detection_model.h5')

def load_ml_model(model_path):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = load_ml_model(f'{working_dir}/diabetes_prediction_model')
heart_disease_model = load_ml_model(f'{working_dir}/heart_deases_prediction_model')

# Function to make predictions
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                     BMI, DiabetesPedigreeFunction, Age):
    
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

    prediction = diabetes_model.predict([user_input])
    return prediction[0]

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    cp_mapping = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    slope_mapping = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    thal_mapping = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2
    }
    fbs_mapping={
        "True":1,
        "False":0
    }
    sex_mapping={
        "Male":1,
        "Female":0
    }
    restecg_mapping={
        "Normal":0,
        "ST-T wave abnormality":1,
        "Left ventricular hypertrophy":2
    }
    exang_mapping={
        "Yes":1,
        "No":0
    }
    
    user_input = pd.DataFrame([{
    'age': age,
    'sex': sex_mapping.get(sex,0),
    'cp': cp_mapping.get(cp, 0),
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_mapping.get(fbs,0),
    'restecg': restecg_mapping.get(restecg,0),
    'thalach': thalach,
    'exang': exang_mapping.get(exang,0),
    'oldpeak': oldpeak,
    'slope': slope_mapping.get(slope, 0),
    'ca': ca,
    'thal': thal_mapping.get(thal, 0)
}])

    prediction = heart_disease_model.predict(user_input)
    return prediction[0]

# Sidebar menu
selected_option = st.sidebar.selectbox(
    "Select Disease Prediction",
    ("Diabetes Prediction", "Heart Disease Prediction","Pneumonia Prediction")
)

# Main content based on selected option
st.title("Multiple Disease Prediction Application")

if selected_option == "Diabetes Prediction":
    st.subheader("Diabetes Prediction 🩸")
    st.write("Fill in the following details to predict diabetes:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose Level", min_value=0.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
        insulin = st.number_input("Insulin", min_value=0.0)
    
    with col2:
        bmi = st.number_input("BMI", min_value=0.0)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                                      bmi, diabetes_pedigree_function, age)
        if prediction == 1:
            st.error("The person has diabetes.")
        else:
            st.success("The person does not have diabetes.")

elif selected_option == "Heart Disease Prediction":
    st.subheader("Heart Disease Prediction 💓")
    st.write("Fill in the following details to predict heart disease:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0.0)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
        restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T wave abnormality", "Probable or definite left ventricular hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0)
    
    with col3:
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, step=1)
        thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])


    if st.button("Predict Heart Disease"):
        prediction = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                                           thalach, exang, oldpeak, slope, ca, thal)
        
        if prediction == 1:
            st.error("The person is predicted to have heart disease.")
        else:
            st.success("The person is predicted to not have heart disease.")


elif selected_option == "Pneumonia Prediction":
    st.subheader("Pneumonia Prediction 🦠")
    st.write("Upload a chest X-ray image to predict pneumonia:")
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = load_img(image_file, target_size=(150, 150))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0

        prediction = model_pneumonia.predict(image)
        st.image(image_file, caption="Uploaded Image:",width=250)

        if prediction[0][0] > 0.5:
            st.error("The person is predicted to have pneumonia.")
        else:
            st.success("The person is predicted to not have pneumonia.")
    