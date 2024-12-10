import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import math

# Load your pre-trained models
best_modelRNN = joblib.load("best_modelRNN.pkl")
best_rf_model = joblib.load("best_rf_model.pkl")

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")  # Ensure you save your scaler as scaler.pkl when training

# Function for prediction
def predict_with_rnn(input_data):
    # Reshape input for the RNN model (samples, timesteps, features)
    input_array = np.array(input_data).reshape(1, 1, -1)
    log_sales_prediction = best_modelRNN.predict(input_array)[0]
    return math.exp(log_sales_prediction)  # Convert log scale back to original scale

def predict_with_rf(input_data):
    # Reshape input for Random Forest model (flat input)
    input_array = np.array(input_data).reshape(1, -1)
    log_sales_prediction = best_rf_model.predict(input_array)[0]
    return math.exp(log_sales_prediction)  # Convert log scale back to original scale

# Streamlit App Layout
st.title("Sales Prediction App")

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", ["RNN Model", "Random Forest Model"])

st.header("Input Features")

# User inputs for the features with min-max constraints based on training data
temperature = st.slider("Temperature (°C)", min_value=-2.06, max_value=100.14, value=60.09)
fuel_price = st.slider("Fuel Price ($ per gallon)", min_value=2.472, max_value=4.468, value=3.361)
cpi = st.slider("Consumer Price Index (CPI)", min_value=126.064, max_value=227.232, value=171.202)
unemployment = st.slider("Unemployment Rate (%)", min_value=3.88, max_value=14.31, value=7.961)
dept = st.slider("Department (Dept ID)", min_value=1, max_value=99, value=44)
size = st.slider("Store Size", min_value=34875, max_value=219622, value=136728)
is_holiday = st.selectbox("Is it a Holiday?", ["No", "Yes"])
type_store = st.selectbox("Store Type", ["A", "B", "C"])

# Convert categorical features
is_holiday = 1 if is_holiday == "Yes" else 0
type_store_mapping = {"A": 0, "B": 1, "C": 2}
type_store = type_store_mapping[type_store]

# Collect inputs into a list
raw_input_data = [temperature, fuel_price, cpi, unemployment, dept, size, is_holiday, type_store]

# Scale the input features using the same scaler used during training
scaled_input_data = scaler.transform([raw_input_data]).flatten()

if st.button("Predict"):
    if selected_model == "RNN Model":
        prediction = predict_with_rnn(scaled_input_data)
    elif selected_model == "Random Forest Model":
        prediction = predict_with_rf(scaled_input_data)

    st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")
