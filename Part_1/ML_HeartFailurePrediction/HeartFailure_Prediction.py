import streamlit as st
import joblib
import pandas as pd

# Streamlit app title
st.title('Predict Heart Failure Based on Health Metrics')

# Upload trained model
uploaded_model = st.file_uploader("Upload your trained model (joblib file)", type=["joblib"])

if uploaded_model is not None:
    # Load the trained model
    model = joblib.load(uploaded_model)

    # Input form for user data
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase Level", min_value=0)
    diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
    ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
    high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
    platelets = st.number_input("Platelets Count", min_value=0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0.0)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])
    time = st.number_input("Time", min_value=0)  # Ensure you include this if it's a feature

    # Create a button to predict
    if st.button("Predict"):
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            "age": [age],
            "anaemia": [anaemia],
            "creatinine_phosphokinase": [creatinine_phosphokinase],
            "diabetes": [diabetes],
            "ejection_fraction": [ejection_fraction],
            "high_blood_pressure": [high_blood_pressure],
            "platelets": [platelets],
            "serum_creatinine": [serum_creatinine],
            "serum_sodium": [serum_sodium],
            "sex": [sex],
            "smoking": [smoking],
            "time": [time]  # Include 'time' if it's used in the model
        })

        # Make predictions
        prediction = model.predict(input_data)
        predicted_probabilities = model.predict_proba(input_data)

        # Display results
        if prediction[0] == 1:
            st.write("Prediction: The patient is likely to have heart failure.")
        else:
            st.write("Prediction: The patient is unlikely to have heart failure.")

        st.write("Probability of Heart Failure:", predicted_probabilities)

else:
    st.warning("Please upload your trained model to make predictions.")
