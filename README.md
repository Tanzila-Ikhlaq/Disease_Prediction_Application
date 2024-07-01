# Diseases Prediction Application

This Streamlit application predicts the likelihood of diabetes, heart disease, and pneumonia based on user input or uploaded images using pre-trained machine learning models.

## Application Overview

The application provides predictions for three types of diseases:

- **Diabetes Prediction**: Predicts whether a person has diabetes based on health metrics.
- **Heart Disease Prediction**: Predicts the likelihood of heart disease based on clinical data.
- **Pneumonia Prediction**: Predicts pneumonia from chest X-ray images.

## Features

- **User Interface**: Interactive interface with sliders and input fields for entering health metrics or uploading X-ray images.
- **Prediction**: Utilizes pre-trained machine learning models (Random Forest for diabetes and heart disease, CNN for pneumonia) to make predictions.
- **Output**: Displays prediction results indicating whether a person is likely to have the disease or not.

## Technologies Used

- Python
- Streamlit
- TensorFlow/Keras (for pneumonia prediction)
- Pandas, NumPy
- Joblib (for model persistence)
- OpenCV (for image processing)

## Files Included

- `app.py`: Main script containing the Streamlit application code.
- `models/`: Directory housing the diabetes and heart disease prediction models.
- `datasets/`: Contains the datasets used for training and evaluation.
- `notebooks/`: Includes Jupyter notebooks for machine learning training and analysis.

## Dataset Used

For pneumonia detection, this project utilizes the Kaggle dataset `tolgadincer/labeled-chest-xray-images`. This dataset includes labeled chest X-ray images used for training and evaluating the pneumonia detection model.

## Screenshots

![Pneumonia Prediction](https://github.com/Tanzila-Ikhlaq/ProjectRisk/assets/141930681/45f3e96d-a102-45d1-a4c6-7eac1b1a0ae0)
*Screenshot of the Pneumonia Prediction interface*

![Diabetes Prediction](https://github.com/Tanzila-Ikhlaq/ProjectRisk/assets/141930681/d9904aa0-8b3d-4fd9-84f9-b662408f2afa)
*Screenshot of the Diabetes Prediction interface*

![Heart Disease Prediction](https://github.com/Tanzila-Ikhlaq/ProjectRisk/assets/141930681/232037be-74bc-49ca-bf03-635d055e9423)
*Screenshot of the Heart Disease Prediction interface*

