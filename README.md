# 🌡️ Urban Heat Island (UHI) Predictor — India

### 👨‍💻 Team: Tech Titans  
Manmath · Krishna · Gunjan · Om  

---

## 📌 Project Overview

This project predicts **Urban Heat Island (UHI) severity** for major Indian cities using:

- 🌐 Live weather data (Open-Meteo API)
- 🤖 Machine Learning (XGBoost Classifier)
- 📊 Interactive visualization (Streamlit)

---

## ⚙️ System Workflow

1. User selects a city
2. API fetches live urban & rural temperature
3. UHI intensity is calculated
4. Features are passed to ML model
5. Model predicts severity:
   - None
   - Mild
   - Moderate
   - Severe

---

## 🧠 Machine Learning Model

### ✔ Features Used
- Temperature
- Elevation
- Population Density
- Energy Consumption
- AQI
- Urban Greenness Ratio
- Wind Speed
- Humidity
- Annual Rainfall

---

### ✔ Model Details
- Algorithm: XGBoost Classifier
- Accuracy: ~100% (rule-based labeling)
- Output: UHI severity classification

---

## 📂 Project Structure
uhi-predictor/
├── config.py
├── pipeline.py
├── app.py
├── data/
├── model/
├── notebooks/
└── docs/


---

## 🚀 How to Run

```bash
git clone <repo-link>
cd uhi-predictor

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py




---

# 🚀 PART 3 — MESSAGE FOR GUNJAN (SEND THIS)

Use this EXACT WhatsApp message:

---

:::writing{variant="chat_message" id="84219"}
Gunjan, I have completed my part (ML model training + full integration with pipeline). The model is now working with live API data and predictions are running successfully.

Now you need to complete the UI (your part). Follow this:

1. Pull latest code from "manmath" branch
2. Go inside uhi-predictor folder
3. Install dependencies:
   pip install streamlit plotly requests
4. Run:
   streamlit run app.py

Your task:
- Build the Streamlit dashboard (use pipeline.py)
- Add city selector (dropdown)
- Show prediction output (severity, temp, UHI)
- Add India map visualization (important for marks)
- Make UI clean and visually appealing

Important:
Just call this function:
from pipeline import predict_uhi

result = predict_uhi("Mumbai")

Display the result — no ML work needed from your side.

If you get stuck for more than 20–30 mins, message immediately.
:::

---

# 🚀 PART 4 — REPO IMPROVEMENTS (VERY IMPORTANT)

I reviewed your structure :contentReference[oaicite:0]{index=0} — here’s what to improve:

---

## 🔥 1. Add `.gitignore` (CRITICAL)

Create `.gitignore`:

```text
venv/
__pycache__/
*.pyc
data/*.csv
model/*.pkl
