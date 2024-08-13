import streamlit as st
import pickle
import numpy as np

# Load the trained model from a pickle file
model_filename = "weather_model.pkl"

@st.cache_resource
def load_model():
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Mapping of encoded summary to descriptive labels
summary_mapping = {
    0: "Clear",
    1: "Foggy",
    2: "Mostly cloudy",
    3: "Overcast",
    4: "Partly cloudy"
}

# Function to predict summary based on user inputs
def predict_summary(weather, temperature, humidity, pressure):
    features = np.array([[weather, temperature, humidity, pressure]])
    prediction = model.predict(features)
    return prediction[0]

# Streamlit UI
st.title("Weather Summary Prediction App")
st.write("Enter the weather features to predict the weather summary.")

# User inputs
weather = st.number_input("Weather (as an integer code)", min_value=0, max_value=10, value=1)
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
pressure = st.number_input("Pressure (millibars)", min_value=900.0, max_value=1100.0, value=1013.0)

if st.button("Predict Summary"):
    summary_code = predict_summary(weather, temperature, humidity, pressure)
    summary_label = summary_mapping.get(summary_code, "Unknown")
    
    st.subheader("Predicted Weather Summary")
    st.markdown(f"<h4 style='color: blue;'>{summary_label}</h2>", unsafe_allow_html=True)
