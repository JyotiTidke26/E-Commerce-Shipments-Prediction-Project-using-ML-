import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, train columns
try:
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/train_columns.pkl", "rb") as f:
        train_columns = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Define numeric and categorical columns
numeric_features = train_columns[:5]  # first 5 are numeric
categorical_features = train_columns[5:]  # last 4 are categorical dummies

# Streamlit UI
st.set_page_config(page_title="E-Commerce Delivery Prediction", layout="centered")
st.title("🚚 E-Commerce Shipments Delivery Prediction ")
st.markdown("Enter shipment and product details to predict if delivery will be on time or not.")

# Inputs
customer_rating = st.slider("Customer Rating (1-5)", 1, 5, 3)
cost_of_product = st.number_input("Cost of the Product ($)", min_value=0.0, value=100.0)
discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, value=5.0)
weight_in_gms = st.number_input("Weight of Product (grams)", min_value=0.0, value=5000.0)
if weight_in_gms == 0:
    st.warning("Weight of product cannot be zero. Please enter a valid value.")
cost_per_gm = cost_of_product / weight_in_gms if weight_in_gms != 0 else 0

mode_of_shipment = st.selectbox("Mode of Shipment", ["Road", "Ship", "Flight"])
product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])

# Encode categorical
mode_road = 1 if mode_of_shipment == "Road" else 0
mode_ship = 1 if mode_of_shipment == "Ship" else 0
importance_low = 1 if product_importance == "low" else 0
importance_medium = 1 if product_importance == "medium" else 0

# Create input dataframe with all train columns
input_dict = {
    "Customer_rating": customer_rating,
    "Cost_of_the_Product": cost_of_product,
    "Discount_offered": discount_offered,
    "Weight_in_gms": weight_in_gms,
    "Cost_per_gm": cost_per_gm,
    "Mode_of_Shipment_Road": mode_road,
    "Mode_of_Shipment_Ship": mode_ship,
    "Product_importance_low": importance_low,
    "Product_importance_medium": importance_medium,
}

input_data = pd.DataFrame([input_dict])
input_data = input_data.reindex(columns=train_columns, fill_value=0)

# Split numeric and categorical
input_numeric = input_data[numeric_features]
input_categorical = input_data[categorical_features]

# Scale numeric features only
scaled_numeric = scaler.transform(input_numeric)

# Combine scaled numeric and categorical
scaled_input = np.hstack([scaled_numeric, input_categorical.values])

# Predict function
def predict_delay(input_arr):
    pred = model.predict(input_arr)[0]
    prob = model.predict_proba(input_arr)[0][1]
    return pred, prob

# Show inputs
st.markdown("### Your Inputs")
st.write(input_data)

# On button click, predict
if st.button("Predict Delivery Status"):
    with st.spinner("Predicting..."):
        prediction, probability = predict_delay(scaled_input)
    if prediction == 1:
        st.error(f"⚠️ Delivery likely delayed (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Delivery likely on time (Confidence: {1 - probability:.2f})")