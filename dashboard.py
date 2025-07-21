import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from PIL import Image
import io
import os

# --- CONFIG ---
OPENWEATHER_API_KEY = "c1b5f8e67322db6354ffa0e332c2d817"
WEATHER_ICON_URL = "https://openweathermap.org/img/wn/{}@2x.png"

st.set_page_config(page_title="Trash Detection Dashboard", layout="wide")
st.title("🌊 Underwater Trash Detection Dashboard")

# --- DATA LOADING ---
st.sidebar.header("Upload Detection Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV with trash detections", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "uploaded file"
elif os.path.exists("detections_sample.csv"):
    df = pd.read_csv("detections_sample.csv")
    st.info("No file uploaded. Loaded sample data from detections_sample.csv.")
    source = "sample file"
else:
    st.error("Please upload a CSV file or add detections_sample.csv to the project directory.")
    st.stop()

# Parse date
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# --- FILTERS ---
st.sidebar.header("Filters")
min_date, max_date = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
trash_types = df['trash_type'].unique().tolist()
selected_types = st.sidebar.multiselect("Trash Type", trash_types, default=trash_types)
severity_levels = sorted(df['severity'].unique())
selected_severity = st.sidebar.slider("Severity Level", min_value=int(min(severity_levels)), max_value=int(max(severity_levels)), value=(int(min(severity_levels)), int(max(severity_levels))))

# Apply filters
mask = (
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1])) &
    (df['trash_type'].isin(selected_types)) &
    (df['severity'] >= selected_severity[0]) &
    (df['severity'] <= selected_severity[1])
)
df_filtered = df[mask]

# --- 1. TRASH DETECTION HEATMAP ---
st.header("🗺️ Trash Detection Heatmap")
if not df_filtered.empty:
    m = folium.Map(location=[df_filtered['latitude'].mean(), df_filtered['longitude'].mean()], zoom_start=5)
    heat_data = df_filtered[['latitude', 'longitude', 'severity']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.3, max_zoom=1).add_to(m)
    st_folium(m, width=700, height=400)
else:
    st.warning("No data to display on heatmap with current filters.")

# --- 2. DETECTION HISTORY TIMELINE ---
st.header("🕒 Detection History Timeline")
if not df_filtered.empty:
    timeline_data = []
    for idx, row in df_filtered.iterrows():
        entry = {
            "Time": row['date'].strftime('%Y-%m-%d %H:%M'),
            "Location": f"({row['latitude']:.2f}, {row['longitude']:.2f})",
            "Type": row['trash_type'],
            "Severity": row['severity'],
            "Image": row['image_path'] if 'image_path' in row and pd.notna(row['image_path']) else None
        }
        timeline_data.append(entry)
    fig = go.Figure()
    for i, entry in enumerate(timeline_data):
        fig.add_trace(go.Scatter(
            x=[entry['Time']],
            y=[entry['Severity']],
            mode='markers+text',
            marker=dict(size=16, color='royalblue'),
            text=[f"{entry['Type']}<br>{entry['Location']}"],
            textposition="top center",
            name=entry['Type'],
            hovertext=f"{entry['Type']}<br>{entry['Location']}<br>Severity: {entry['Severity']}"
        ))
    fig.update_layout(
        title="Detection Timeline (Severity as Y)",
        xaxis_title="Time",
        yaxis_title="Severity",
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No detection history to show with current filters.")

# --- 3. WEATHER FORECAST FOR CLEANUP ---
st.header("🌦️ Weather Forecast for Cleanup")
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'list' in data:
            return data['list'][:8]  # Next 24 hours (3h intervals)
        else:
            return []
    except Exception as e:
        return []

for idx, row in df_filtered.iterrows():
    st.subheader(f"Location: ({row['latitude']:.2f}, {row['longitude']:.2f}) | Type: {row['trash_type']} | Severity: {row['severity']}")
    weather_data = get_weather(row['latitude'], row['longitude'])
    if weather_data:
        cols = st.columns(len(weather_data))
        for i, w in enumerate(weather_data):
            with cols[i]:
                icon_url = WEATHER_ICON_URL.format(w['weather'][0]['icon'])
                st.image(icon_url, width=50)
                st.write(f"{w['dt_txt'][11:16]}")
                st.write(f"{w['main']['temp']}°C")
                st.write(w['weather'][0]['main'])
    else:
        st.info("Weather data unavailable for this location.")
    # --- 4. PLAN CLEANUP BUTTON ---
    if st.button(f"Plan Cleanup for ({row['latitude']:.2f}, {row['longitude']:.2f})", key=f"plan_{idx}"):
        # Find best weather window (no rain, mild temp)
        best = None
        for w in weather_data:
            if w['weather'][0]['main'] not in ['Rain', 'Thunderstorm'] and 15 <= w['main']['temp'] <= 30:
                best = w
                break
        if best:
            st.success(f"Ideal cleanup window: {best['dt_txt']} ({best['main']['temp']}°C, {best['weather'][0]['main']})")
        else:
            st.warning("No ideal weather window found in next 24 hours.")
    st.markdown("---")

st.caption("Dashboard by AI | Data updates in real-time with CSV upload. Mobile-friendly and interactive.") 