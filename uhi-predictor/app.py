# Main Streamlit Application UI

import streamlit as st
import pandas as pd
import plotly.express as px
from config import CITIES, SEVERITY_COLORS
from pipeline import MODEL_PATH, predict_uhi
import os

# Set page config
st.set_page_config(page_title="UHI Predictor", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)
# CSS for severity badge
st.markdown("""
<style>
.severity-badge {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Top Bar
st.title("🌆 Urban Heat Island (UHI) Predictor")
st.caption("Real-time AI-powered environmental heat analysis across Indian cities")
st.divider()
# Model Status Banner
if os.path.exists(MODEL_PATH):
    st.success("🤖 ML Model active — XGBoost classifier loaded", icon="✅")
else:
    st.warning("Preview mode — waiting for model file. App works with rule-based fallback until Krishna uploads uhi_model.pkl", icon="⚠️")

# Sidebar
with st.sidebar:
    st.header("Controls")
    selected_city = st.selectbox("Select City", options=list(CITIES.keys()))
    predict_btn = st.button("Predict UHI", type="primary", use_container_width=True)
    
    st.divider()
    st.subheader("Severity Legend")
    for level, color in SEVERITY_COLORS.items():
        st.markdown(f'<div style="background-color: {color}; padding: 5px 10px; border-radius: 5px; color: white; margin-bottom: 5px;">{level}</div>', unsafe_allow_html=True)
        
    st.divider()
    st.caption("Live data: Open-Meteo API | Model: XGBoost")

# Initialize session state for map data
if "map_data" not in st.session_state:
    st.session_state.map_data = []

# Main Layout: 2 sections (Map and Single City)
tab1, tab2 = st.tabs(["🌍 India Live Map", "🏢 Single City Result"])

with tab1:
    st.header("India Live UHI Map")
    if st.button("Load live map"):
        with st.spinner("Fetching live data for all cities..."):
            all_cities_data = []
            for city in CITIES.keys():
                res = predict_uhi(city)
                if res is not None:
                    lat, lon = CITIES[city]["urban"]
                    res["lat"] = lat
                    res["lon"] = lon
                    all_cities_data.append(res)
            st.session_state.map_data = all_cities_data
            
    if st.session_state.map_data:
        df = pd.DataFrame(st.session_state.map_data)
        
        # Plotly map
        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            hover_name="city",
            hover_data={
                "lat": False,
                "lon": False,
                "urban_temp": True,
                "rural_temp": True,
                "uhi_intensity": True,
                "severity": True
            },
            size="uhi_intensity",
            color="uhi_intensity",
            color_continuous_scale=["cyan", "green", "orange", "red", "darkred"],
            range_color=[0, 6],
            text="city"
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        fig.update_geos(
            scope="asia",
            center=dict(lat=22, lon=80),
            projection_scale=4,
            landcolor="#1e293b",
            oceancolor="#0f172a",
            bgcolor="#0f172a",
            showcountries=True,
            countrycolor="#475569"
        )
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="#0f172a",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("City Data")
        display_df = df[["city", "urban_temp", "rural_temp", "uhi_intensity", "severity"]].copy()
        display_df.columns = ["City", "Urban Temp (°C)", "Rural Temp (°C)", "UHI Intensity (°C)", "Severity"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    if predict_btn:
        with st.spinner(f"Fetching live data and predicting UHI for {selected_city}..."):
            res = predict_uhi(selected_city)
            
            if res is None:
                st.error("Failed to fetch live data from the API. Please try again later.")
            else:
                # Top section: Cards
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Urban Temp", f"{res['urban_temp']} °C")
                col2.metric("Rural Temp", f"{res['rural_temp']} °C")
                diff = res['uhi_intensity']
                col3.metric("UHI Intensity", f"{diff} °C", f"{diff} (difference)" if diff > 0 else "")
                col4.metric("Humidity", f"{res['humidity']} %")
                
                # Severity Badge
                st.markdown(f'<div class="severity-badge" style="background-color: {res["color"]};">Severity: {res["severity"]}</div>', unsafe_allow_html=True)
                if res["severity"] == "Severe":
                    st.error("🔥 Extreme Urban Heat — Stay Indoors")
                elif res["severity"] == "Moderate":
                    st.warning("⚠ Moderate Heat Island Effect")
                elif res["severity"] == "Mild":
                    st.info("🌤 Mild Heat Conditions")
                else:
                    st.success("✅ Safe Temperature Conditions")
                # Health Advisory
                if res["severity"] == "Severe":
                    st.error("Health Advisory: Extreme heat conditions. Vulnerable populations should stay indoors. High risk of heatstroke.")
                elif res["severity"] == "Moderate":
                    st.warning("Health Advisory: Significant heat differences detected. Ensure adequate hydration and limit outdoor exertion.")
                elif res["severity"] == "Mild":
                    st.info("Health Advisory: Mild heat island effect. Conditions are generally manageable, but stay mindful of hydration.")
                else:
                    st.success("Health Advisory: Favorable thermal conditions. No significant heat concerns.")
                
                # Technical Expander
                with st.expander("Technical details"):
                    from config import FEATURE_COLUMNS
                    
                    st.write(f"**ML Model Used:** {'Yes (XGBoost)' if res['using_ml_model'] else 'No (Rule-based Fallback)'}")
                    
                    st.write("**Feature Vector Matrix:**")
                    # Display the exact features used for prediction
                    feature_dict = dict(zip(FEATURE_COLUMNS, res['features']))
                    st.json(feature_dict)

    elif not predict_btn:
        st.info("Select a city from the sidebar and click 'Predict UHI' to see results.")

st.divider()
st.caption("Live data: Open-Meteo API | Training: Kaggle UHI Dataset | Model: XGBoost Classifier | Team Tech Titans 2026")
