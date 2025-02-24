# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
# from sklearn.linear_model import LogisticRegression

# Load the trained model
@st.cache_resource
def load_model():
    # Load your trained model here
    # For example, if you saved it as a pickle file:
    with open('breast_cancer_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict diagnosis
def predict_diagnosis(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Breast Cancer Classification", page_icon="🩺", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🩺 Breast Cancer Classification")
    st.markdown("This app predicts whether a breast cancer diagnosis is **Malignant (M)** or **Benign (B)** based on input features.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Your Name", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a Logistic Regression model to classify breast cancer diagnoses as Malignant (0) or Benign (1).")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    st.subheader("Enter the features for prediction:")
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, value=14.0, help="Mean radius of the tumor")
        texture_mean = st.number_input("Texture Mean", min_value=0.0, value=19.0, help="Mean texture of the tumor")
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=91.0, help="Mean perimeter of the tumor")
        area_mean = st.number_input("Area Mean", min_value=0.0, value=650.0, help="Mean area of the tumor")
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1, help="Mean smoothness of the tumor")
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.1, help="Mean compactness of the tumor")
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.1, help="Mean concavity of the tumor")
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.05, help="Mean concave points of the tumor")
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2, help="Mean symmetry of the tumor")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.06, help="Mean fractal dimension of the tumor")

    with col2:
        radius_se = st.number_input("Radius SE", min_value=0.0, value=0.5, help="Standard error of the radius")
        texture_se = st.number_input("Texture SE", min_value=0.0, value=1.0, help="Standard error of the texture")
        perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=2.0, help="Standard error of the perimeter")
        area_se = st.number_input("Area SE", min_value=0.0, value=40.0, help="Standard error of the area")
        smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.01, help="Standard error of the smoothness")
        compactness_se = st.number_input("Compactness SE", min_value=0.0, value=0.05, help="Standard error of the compactness")
        concavity_se = st.number_input("Concavity SE", min_value=0.0, value=0.03, help="Standard error of the concavity")
        concave_points_se = st.number_input("Concave Points SE", min_value=0.0, value=0.01, help="Standard error of the concave points")
        symmetry_se = st.number_input("Symmetry SE", min_value=0.0, value=0.02, help="Standard error of the symmetry")
        fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, value=0.01, help="Standard error of the fractal dimension")

    with col3:
        radius_worst = st.number_input("Radius Worst", min_value=0.0, value=16.0, help="Worst radius of the tumor")
        texture_worst = st.number_input("Texture Worst", min_value=0.0, value=25.0, help="Worst texture of the tumor")
        perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=107.0, help="Worst perimeter of the tumor")
        area_worst = st.number_input("Area Worst", min_value=0.0, value=880.0, help="Worst area of the tumor")
        smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.15, help="Worst smoothness of the tumor")
        compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.25, help="Worst compactness of the tumor")
        concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.3, help="Worst concavity of the tumor")
        concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.1, help="Worst concave points of the tumor")
        symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.3, help="Worst symmetry of the tumor")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=0.1, help="Worst fractal dimension of the tumor")

    # Prepare input data for the model
    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
        concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
        concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]

    # Prediction button
    if st.button("Predict Diagnosis"):
        try:
            prediction = predict_diagnosis(input_data)
            
            if prediction == 0:
                result_placeholder.error("Prediction: **Malignant (M)**")
            else:
                result_placeholder.success("Prediction: **Benign (B)**")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()