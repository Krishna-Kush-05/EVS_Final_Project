# Pipeline logic for data fetching, processing, and model prediction.

import os
import pickle
import requests
import streamlit as st
import numpy as np
from config import CITIES, FEATURE_COLUMNS, SEVERITY_COLORS, SEVERITY_LABELS

MODEL_PATH = "model/uhi_model.pkl"

def load_model():
    """Load the trained XGBoost model if it exists, else return None for rule-based fallback."""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("ML model loaded — XGBoost classifier active")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("WARNING: uhi_model.pkl not found — using rule-based fallback")
        return None

# Load model once at module level
MODEL = load_model()

@st.cache_data(ttl=1800)
def fetch_live_temps(lat, lon):
    """
    Fetch live temperature, humidity, and wind speed from Open-Meteo API.
    Returns (temperature_2m, relativehumidity_2m, windspeed_10m)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "Asia/Kolkata"
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        
        temp = current.get("temperature_2m")
        hum = current.get("relativehumidity_2m")
        wind = current.get("windspeed_10m")
        
        # If any of the features are missing, return Nones
        if temp is None or hum is None or wind is None:
            return None, None, None
            
        return temp, hum, wind
    except Exception as e:
        # Handle API errors gracefully
        print(f"Error fetching live data: {e}")
        return None, None, None

def predict_severity(features, model):
    """
    Predict severity using the model if available, else use a rule-based fallback.
    Severity labeling logic (used by Krishna in Colab):
    - temp >= 38 and green <= 20 -> 3 (Severe)
    - temp >= 34 and green <= 35 -> 2 (Moderate)
    - temp >= 30 and green <= 50 -> 1 (Mild)
    - else -> 0 (None)
    """
    if model is not None:
        return int(model.predict(np.array([features]))[0])
    else:
        # Rule-based fallback using Temperature (index 0) and Urban Greenness Ratio (index 5)
        temp = features[0]
        # In FEATURE_COLUMNS, index 5 is Urban Greenness Ratio
        green = features[5] 
        
        if temp >= 38 and green <= 20:   return 3
        elif temp >= 34 and green <= 35: return 2
        elif temp >= 30 and green <= 50: return 1
        else:                            return 0

def predict_uhi(city_name):
    """
    Predict UHI intensity and severity for a given city.
    """
    if city_name not in CITIES:
        return None
        
    cfg = CITIES[city_name]
    
    # Get urban coords
    urban_lat, urban_lon = cfg["urban"]
    urban_temp, humidity, wind = fetch_live_temps(urban_lat, urban_lon)
    
    # Get rural coords
    rural_lat, rural_lon = cfg["rural"]
    rural_temp, _, _ = fetch_live_temps(rural_lat, rural_lon)
    
    if urban_temp is None or rural_temp is None:
        return None
        
    # UHI Intensity (urban temp - rural temp)
    uhi_intensity = round(urban_temp - rural_temp, 2)
    
    # Build feature vector according to FEATURE_COLUMNS order
    features = [
        urban_temp,                           # live from API
        cfg["Elevation"],                     # fixed
        cfg["Population Density"],            # fixed
        cfg["Energy Consumption"],            # fixed
        cfg["AQI"],                           # fixed
        cfg["Urban Greenness Ratio"],         # fixed
        wind,                                 # live from API
        humidity,                             # live from API
        cfg["Annual Rainfall"],               # fixed
    ]
    
    # Predict severity
    severity_idx = predict_severity(features, MODEL)
    severity_label = SEVERITY_LABELS.get(severity_idx, "Unknown")
    color = SEVERITY_COLORS.get(severity_label, "#94a3b8")
    
    return {
        "city": city_name,
        "urban_temp": round(urban_temp, 1),
        "rural_temp": round(rural_temp, 1),
        "uhi_intensity": uhi_intensity,
        "humidity": humidity,
        "wind": wind,
        "features": features,
        "severity": severity_label,
        "severity_code": severity_idx,
        "color": color,
        "using_ml_model": MODEL is not None
    }
