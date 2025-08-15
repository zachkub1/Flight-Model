import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.title("✈️ Flight Price Predictor")
st.markdown("Predict flight prices based on various factors like airline, route, time, and class.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load a saved model first
        with open('flight_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except:
        # If no saved model, return None
        return None, False

# Load model
model, model_loaded = load_model()

if not model_loaded:
    st.warning("⚠️ No trained model found. Please run main.py first to train the model.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("🛫 Flight Details")

# Airline selection
airlines = ['Air India', 'AirAsia', 'GO FIRST', 'IndiGo', 'SpiceJet', 'Vistara']
selected_airline = st.sidebar.selectbox("Select Airline", airlines)

# Source and destination cities
cities = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
source_city = st.sidebar.selectbox("From", cities)
destination_city = st.sidebar.selectbox("To", cities)

# Time selections
departure_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
arrival_times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']

departure_time = st.sidebar.selectbox("Departure Time", departure_times)
arrival_time = st.sidebar.selectbox("Arrival Time", arrival_times)

# Other parameters
stops = st.sidebar.selectbox("Number of Stops", [0, 1, 2])
flight_class = st.sidebar.selectbox("Class", ['Economy', 'Business'])
duration = st.sidebar.slider("Duration (hours)", 1.0, 10.0, 2.0, 0.5)

# Convert class to numeric
class_numeric = 1 if flight_class == 'Business' else 0

# Create feature vector
def create_features():
    # Initialize all features to 0
    features = {}
    
    # Airline features
    for airline in airlines:
        features[f'airline_{airline}'] = 1 if airline == selected_airline else 0
    
    # Source city features
    for city in cities:
        features[f'source_city_{city}'] = 1 if city == source_city else 0
    
    # Destination city features
    for city in cities:
        features[f'destination_city_{city}'] = 1 if city == destination_city else 0
    
    # Departure time features
    for time in departure_times:
        features[f'departure_{time}'] = 1 if time == departure_time else 0
    
    # Arrival time features
    for time in arrival_times:
        features[f'arrival_{time}'] = 1 if time == arrival_time else 0
    
    # Other features
    features['stops'] = stops
    features['class'] = class_numeric
    features['duration'] = duration
    
    return features

# Prediction function
def predict_price():
    features = create_features()
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Ensure column order matches training data
    if hasattr(model, 'feature_names_in_'):
        feature_df = feature_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    return prediction

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Flight Information Summary")
    
    # Display selected parameters
    st.write(f"**Airline:** {selected_airline}")
    st.write(f"**Route:** {source_city} → {destination_city}")
    st.write(f"**Departure:** {departure_time}")
    st.write(f"**Arrival:** {arrival_time}")
    st.write(f"**Stops:** {stops}")
    st.write(f"**Class:** {flight_class}")
    st.write(f"**Duration:** {duration} hours")
    
    # Predict button
    if st.button("🚀 Predict Price", type="primary"):
        with st.spinner("Calculating price..."):
            try:
                predicted_price = predict_price()
                
                # Display result
                st.success(f"**Predicted Price: ₹{predicted_price:,.2f}**")
                
                # Price range indicator
                if predicted_price < 5000:
                    st.info("💡 This appears to be a budget-friendly option!")
                elif predicted_price < 15000:
                    st.info("💡 This is a mid-range flight option.")
                else:
                    st.info("💡 This is a premium flight option.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

