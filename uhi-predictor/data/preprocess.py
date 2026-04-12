import pandas as pd

df = pd.read_csv("data/urban_heat_island_dataset.csv")

# Drop missing values
df = df.dropna()

# Rename columns properly (based on YOUR dataset)
df = df.rename(columns={
    "Temperature (°C)": "Temperature",
    "Elevation (m)": "Elevation",
    "Population Density (people/km²)": "Population Density",
    "Energy Consumption (kWh)": "Energy Consumption",
    "Air Quality Index (AQI)": "AQI",
    "Urban Greenness Ratio (%)": "Urban Greenness Ratio",
    "Wind Speed (km/h)": "Wind Speed",
    "Humidity (%)": "Humidity",
    "Annual Rainfall (mm)": "Annual Rainfall"
})

# Create severity label (as per README)
def assign_severity(row):
    temp = row["Temperature"]
    green = row["Urban Greenness Ratio"]

    if temp >= 38 and green <= 20:
        return 3
    elif temp >= 34 and green <= 35:
        return 2
    elif temp >= 30 and green <= 50:
        return 1
    else:
        return 0

df["severity"] = df.apply(assign_severity, axis=1)

# Select ONLY required columns (IMPORTANT ORDER)
df = df[[
    "Temperature",
    "Elevation",
    "Population Density",
    "Energy Consumption",
    "AQI",
    "Urban Greenness Ratio",
    "Wind Speed",
    "Humidity",
    "Annual Rainfall",
    "severity"
]]

# Save processed file
df.to_csv("data/training_ready.csv", index=False)

print("✅ Preprocessing Done")
print(df.head())
print(df["severity"].value_counts())
