import numpy as np
import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib
import math

# Load pre-trained models
best_modelRNN = joblib.load("best_modelRNN.pkl")
best_rf_model = joblib.load("random_forest_model.pkl")

# Load the pre-trained scaler
scaler = joblib.load("scaler.pkl")

# Prediction Functions
def predict_with_rnn(input_data):
    # Reshape input for the RNN model (samples, timesteps, features)
    input_array = np.array(input_data).reshape(1, 1, -1)  # (1, 1, n_features)
    log_sales_prediction = best_modelRNN.predict(input_array)[0]
    return math.exp(log_sales_prediction)  # Convert log scale back to original scale

def predict_with_rf(input_data):
    # Reshape input for Random Forest model (flat input)
    input_array = np.array(input_data).reshape(1, -1)  # (1, n_features)
    log_sales_prediction = best_rf_model.predict(input_array)[0]
    return math.exp(log_sales_prediction)  # Convert log scale back to original scale

# Streamlit App
st.title("Sales Prediction App")

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", ["RNN Model", "Random Forest Model"])

st.header("Input Features")

# User Inputs
store = st.slider("Store ID", min_value=1, max_value=45, value=22)
temperature = st.slider("Temperature (°C)", min_value=-2.06, max_value=100.14, value=60.09)
fuel_price = st.slider("Fuel Price ($ per gallon)", min_value=2.472, max_value=4.468, value=3.361)
cpi = st.slider("Consumer Price Index (CPI)", min_value=126.064, max_value=227.232, value=171.202)
unemployment = st.slider("Unemployment Rate (%)", min_value=3.88, max_value=14.31, value=7.961)
dept = st.slider("Department ID", min_value=1, max_value=99, value=44)
is_holiday = st.selectbox("Is it a Holiday?", ["No", "Yes"])
type_store = st.selectbox("Store Type", ["A", "B", "C"])
size = st.slider("Store Size", min_value=34875, max_value=219622, value=136728)

# Dynamic Features
today = datetime.date.today()
year = st.slider("Year", min_value=2020, max_value=2022, value=today.year)
month = st.slider("Month", min_value=1, max_value=12, value=today.month)
day = st.slider("Day", min_value=1, max_value=31, value=today.day)
weekday = today.weekday()
is_weekend = 1 if weekday in [5, 6] else 0

# Convert categorical inputs
is_holiday = 1 if is_holiday == "Yes" else 0
type_store_mapping = {"A": 0, "B": 1, "C": 2}
type_store = type_store_mapping[type_store]

# Input Features in Correct Order
raw_input_data = [
    store, temperature, fuel_price, cpi, unemployment,
    dept, is_holiday, type_store, size,
    year, month, day, weekday, is_weekend
]

# Debugging: Check input length and order
if len(raw_input_data) != 14:
    st.error(f"Input data must have 14 features, but {len(raw_input_data)} provided.")
else:
    try:
        # Scale the input features
        scaled_input_data = scaler.transform([raw_input_data])  # Ensure input is 2D (1, n_features)
        scaled_input_data = scaled_input_data.flatten()  # Flatten for RandomForest
        
        if st.button("Predict"):
            # Predictions
            if selected_model == "RNN Model":
                prediction = predict_with_rnn(scaled_input_data)  # RNN needs (1, 1, n_features)
            elif selected_model == "Random Forest Model":
                prediction = predict_with_rf(scaled_input_data)  # RandomForest needs (1, n_features)

            st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")

    except ValueError as e:
        st.error(f"Scaling error: {e}")
