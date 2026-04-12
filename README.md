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
uhi-predictor/
├── config.py
├── pipeline.py
├── app.py
│
├── data/
│   ├── preprocess.py
│   └── training_ready.csv
│
├── model/
│   ├── train_model.py
│   └── uhi_model.pkl
│
├── notebooks/
└── docs/
```
---

## 🚀 Getting Started

### 1. Clone Repository

git clone https://github.com/Krishna-Kush-05/EVS_Final_Project.git
cd EVS_Final_Project/uhi-predictor

---

### 2. Setup Virtual Environment

python -m venv venv
venv\Scripts\activate

---

### 3. Install Dependencies

pip install -r requirements.txt

---

### 4. Run Application

streamlit run app.py

---

## 🔍 Example Output

City: Mumbai
UHI Intensity: +0.8°C
Severity: None

---

## 🧩 Key Features

✔ Real-time prediction using live API data
✔ Machine Learning-based classification
✔ Modular and scalable architecture
✔ Ready for visualization and deployment

---

## 📊 Data Sources

* Urban Heat Island Dataset (Kaggle)
* Open-Meteo Weather API

---

## 👥 Team Contributions

Manmath → Pipeline, API integration, system design
Krishna → Data preprocessing, ML model training
Gunjan → Streamlit UI and visualization
Om → EDA, evaluation, reporting

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
