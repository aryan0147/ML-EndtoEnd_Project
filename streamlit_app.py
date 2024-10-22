import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Ridge model and scaler
ridge = pickle.load(open('project/models/ridge.pkl', 'rb'))
scaler = pickle.load(open('project/models/scaler.pkl', 'rb'))

# Streamlit application
st.title("Data Prediction App")

st.markdown("### Input the parameters to get predictions")

# Input widgets for user input
Temperature = st.number_input("Temperature", value=0.0, format="%.2f")
RH = st.number_input("Relative Humidity (RH)", value=0.0, format="%.2f")
Ws = st.number_input("Wind Speed (Ws)", value=0.0, format="%.2f")
Rain = st.number_input("Rain", value=0.0, format="%.2f")
FFMC = st.number_input("FFMC", value=0.0, format="%.2f")
DMC = st.number_input("DMC", value=0.0, format="%.2f")
ISI = st.number_input("ISI", value=0.0, format="%.2f")
Classes = st.number_input("Classes", value=0.0, format="%.2f")
Region = st.number_input("Region", value=0.0, format="%.2f")

# Predict button
if st.button("Predict"):
    # Prepare and scale input data
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    new_data_scaled = scaler.transform(input_data)

    # Make prediction using the Ridge model
    result = ridge.predict(new_data_scaled)

    # Display the result
    st.success(f"Predicted Result: {round(result[0], 2)}")
