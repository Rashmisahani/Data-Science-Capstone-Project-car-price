import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load car data for brand selection
car_data_cleaned = pd.read_csv('car_data_cleaned.csv')  # Adjust path to your data file

# Define categories based on your training data
fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
seller_types = ['Dealer', 'Individual']
transmissions = ['Manual', 'Automatic']
owners = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
brands = car_data_cleaned['brand'].unique()

# Streamlit app
st.title("Car Price Prediction")

year = st.number_input("Year", 1990, 2022)
km_driven = st.number_input("Kilometers Driven", 0, 1000000)
fuel = st.selectbox("Fuel Type", fuel_types)
seller_type = st.selectbox("Seller Type", seller_types)
transmission = st.selectbox("Transmission Type", transmissions)
owner = st.selectbox("Owner Type", owners)
brand = st.selectbox("Car Brand", brands)

# Encoding user input
# Assuming you used integer encoding, replace this part with the appropriate encoding method used during training
fuel_encoded = fuel_types.index(fuel)
seller_type_encoded = seller_types.index(seller_type)
transmission_encoded = transmissions.index(transmission)
owner_encoded = owners.index(owner)
brand_encoded = np.where(brands == brand)[0][0]

input_data = np.array([
    [
        year,
        km_driven,
        fuel_encoded,
        seller_type_encoded,
        transmission_encoded,
        owner_encoded,
        brand_encoded
    ]
])
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted Selling Price: {prediction[0]:.2f}")
