# 🌡️ Urban Heat Island (UHI) Predictor — India

<p align="center">
  <b>AI-powered real-time prediction of Urban Heat Island intensity across Indian cities</b>
</p>

---

## 📖 Overview

Urban Heat Island (UHI) is a major environmental issue where urban areas experience higher temperatures than surrounding rural regions due to human activities and infrastructure.

This project presents a **real-time UHI prediction system** that integrates:

* 🌐 Live weather data using Open-Meteo API
* 🤖 Machine Learning using XGBoost Classifier
* 📊 Interactive visualization (Streamlit dashboard)

---

## 🌍 Live Dashboard

**Access the interactive UHI Predictor dashboard here:**

🔗 **[https://uhi-predictor-evs-project.streamlit.app/]**


---

## ⚙️ System Workflow

User selects city
→ Live weather data fetched from API
→ Feature vector generated
→ Trained ML model predicts UHI severity
→ Result displayed to user

---

## 🧠 Machine Learning Pipeline

### 🔹 Features Used

* Temperature
* Elevation
* Population Density
* Energy Consumption
* AQI
* Urban Greenness Ratio
* Wind Speed
* Humidity
* Annual Rainfall

---

### 🔹 Model Details

* Algorithm: XGBoost Classifier
* Training: Rule-based label generation
* Output Classes:

  * 0 → None
  * 1 → Mild
  * 2 → Moderate
  * 3 → Severe

Note: High accuracy (~100%) is expected because labels are generated using deterministic rules.

---

## 📂 Project Structure

```text
EVS_Final_Project/
├── README.md
├── requirements.txt
│
├── data/
│   ├── preprocess.py
│   └── training_ready.csv
│
├── model/
│   ├── train_model.py
│   └── uhi_model.pkl
│
├── uhi-predictor/
│   ├── config.py
│   ├── pipeline.py
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── notebooks/
│   └── EDA & Analysis.ipynb
│
└── docs/
    └── Technical Documentation
---

## 👥 Team Contributions

Manmath → Pipeline, API integration, system design
Krishna → Data preprocessing, ML model training
Gunjan → Streamlit UI and visualization
Om → EDA, evaluation, reporting

---

## 🖌️ What Gunjan Needs to Do Next (Streamlit Frontend)

Gunjan, your main task is to improve and expand the dashboard UI in `app.py`. The backend logic and ML model are completed and integrated. Please focus on:

### 1. Enhance Visual Aesthetics
* Improve the custom CSS to create a more premium dashboard look.
* Ensure color codes for `Severity` labels strictly match `SEVERITY_COLORS` in `config.py`.

### 2. Interactive Components
* Upgrade the Plotly India Live Map to include better hover animations and potentially a satellite basemap.
* Add supplementary charts (e.g., bar charts comparing urban vs. rural temperature for all cities).

### 3. User Experience (UX)
* Optimize the layout for both wide screens and smaller display windows.
* Expand the **Health Advisory** alerts to include specific actionable items for the user!

*(Note: Please do not change the ML feature order in `config.py` or `pipeline.py`! The pipeline integration is finalized.)*

---

## 🚀 Future Scope

* Integration of satellite data (NDVI)
* Advanced ML / Deep Learning models
* Mobile-friendly interface
* More city coverage

---

## 📌 Conclusion

This project demonstrates how machine learning combined with real-time environmental data can help monitor Urban Heat Islands and support sustainable urban planning.

---

<p align="center">
<b>Built with ❤️ by Team Tech </b>
</p>
