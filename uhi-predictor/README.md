# Urban Heat Island (UHI) Predictor

This is a Streamlit application designed to predict the severity of the Urban Heat Island (UHI) effect for 12 major Indian cities. Currently, it runs on a rule-based fallback system using live weather data fetched from the Open-Meteo API.

---

## What Krishna Needs to Do Next (Machine Learning Pipeline)

Krishna, please follow these exact specifications when training the XGBoost model in Google Colab. The codebase relies on these specific formats to work seamlessly:

### 1. Feature Columns & Order
Your dataset and model inputs **MUST** be in this exact order. Do not change it, or the pipeline predictions will fail/be inaccurate:
1. `Temperature` (°C) — this corresponds to the urban temperature
2. `Elevation` (m)
3. `Population Density` (people/km²)
4. `Energy Consumption` (kWh)
5. `AQI`
6. `Urban Greenness Ratio` (%)
7. `Wind Speed` (km/h)
8. `Humidity` (%)
9. `Annual Rainfall` (mm)

### 2. Severity Target Labels
Since the original dataset lacks an explicit UHI severity column, please use the following logic to generate the `severity` target variables before training:
```python
def assign_severity(row):
    temp = row["Temperature (°C)"]
    green = row["Urban Greenness Ratio (%)"]
    
    if temp >= 38 and green <= 20:   
        return 3  # Severe
    elif temp >= 34 and green <= 35: 
        return 2  # Moderate
    elif temp >= 30 and green <= 50: 
        return 1  # Mild
    else:                            
        return 0  # None
```

### 3. Model Training
Please train an XGBoost classifier with the following hyperparameters (or similar) to ensure consistency:
```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
```

### 4. Exporting and Saving the Model
Once trained, save the model as a pickle file named exactly `uhi_model.pkl`:
```python
import pickle
pickle.dump(model, open('uhi_model.pkl', 'wb'))
```

### 5. Final Step: Integration
Upload/Push `uhi_model.pkl` to the `model/` folder inside this project. 
Once the file is inside the `model/` directory, the UI will automatically switch from **"Preview Mode (Rule-Based Fallback)"** to **"ML Model Active Mode"** without any code changes needed!

---

## Local Setup

To run the application on your own machine:

1. Navigate to the project directory:
   ```bash
   cd uhi-predictor
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit layout:
   ```bash
   streamlit run app.py
   ```
