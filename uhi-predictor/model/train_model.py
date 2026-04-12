import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/training_ready.csv")

X = df[[
    "Temperature",
    "Elevation",
    "Population Density",
    "Energy Consumption",
    "AQI",
    "Urban Greenness Ratio",
    "Wind Speed",
    "Humidity",
    "Annual Rainfall"
]]

y = df["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("model/uhi_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved")
