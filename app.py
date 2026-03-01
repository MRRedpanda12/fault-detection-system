import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import hashlib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Tamil Nadu Fault Detection", layout="wide")

# -------------------------------------------------
# 🔐 LOGIN SYSTEM
# -------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USER = "admin"
PASS_HASH = hash_password("cloud123")

st.sidebar.title("Secure Cloud Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != USER or hash_password(password) != PASS_HASH:
    st.title("🔐 Cloud-Based Fault Detection System")
    st.warning("Please login to access the dashboard.")
    st.stop()

st.sidebar.success("Access Granted ✅")

# -------------------------------------------------
# DATA GENERATION (1500+)
# -------------------------------------------------
np.random.seed(42)

normal_voltage = np.random.normal(230, 4, 1200)
normal_current = np.random.normal(10, 1.5, 1200)

fault_voltage = np.random.normal(140, 20, 350)
fault_current = np.random.normal(28, 8, 350)

voltage = np.concatenate([normal_voltage, fault_voltage])
current = np.concatenate([normal_current, fault_current])

df = pd.DataFrame({
    "Voltage": voltage,
    "Current": current
})

# -------------------------------------------------
# ML MODEL
# -------------------------------------------------
model = IsolationForest(contamination=0.20, random_state=42)
df["Anomaly"] = model.fit_predict(df)
df["Status"] = df["Anomaly"].map({1: "NORMAL", -1: "FAULT"})

# -------------------------------------------------
# PRIORITY LOGIC
# -------------------------------------------------
def assign_priority(row):
    if row["Status"] == "FAULT":
        if row["Current"] > 35:
            return "HIGH"
        elif row["Current"] > 25:
            return "MEDIUM"
        else:
            return "LOW"
    return "NORMAL"

df["Priority"] = df.apply(assign_priority, axis=1)

# -------------------------------------------------
# DASHBOARD METRICS
# -------------------------------------------------
st.title("⚡ Tamil Nadu Wide Fault Detection Dashboard")

total_points = len(df)
total_faults = len(df[df["Status"] == "FAULT"])
high_faults = len(df[df["Priority"] == "HIGH"])
medium_faults = len(df[df["Priority"] == "MEDIUM"])
low_faults = len(df[df["Priority"] == "LOW"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Data", total_points)
col2.metric("Total Faults", total_faults)
col3.metric("High Priority", high_faults)
col4.metric("Medium Priority", medium_faults)
col5.metric("Low Priority", low_faults)

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------
st.subheader("Voltage vs Current Fault Visualization")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Voltage",
    y="Current",
    hue="Priority",
    palette={
        "HIGH": "red",
        "MEDIUM": "orange",
        "LOW": "yellow",
        "NORMAL": "green"
    },
    ax=ax
)
plt.title("Fault Classification with Priority Levels")
st.pyplot(fig)

# -------------------------------------------------
# GIS MAPPING - FIXED VERSION
# -------------------------------------------------
st.subheader("GIS Fault Mapping Across Tamil Nadu")

# Tamil Nadu center
tn_center_lat = 11.1271
tn_center_lon = 78.6569

m = folium.Map(location=[tn_center_lat, tn_center_lon], zoom_start=7)

# Add marker clustering (IMPORTANT FIX)
marker_cluster = MarkerCluster().add_to(m)

fault_points = df[df["Status"] == "FAULT"]

for i in range(len(fault_points)):

    lat = np.random.uniform(8.0, 13.5)
    lon = np.random.uniform(76.0, 80.5)

    priority = fault_points.iloc[i]["Priority"]

    if priority == "HIGH":
        color = "red"
    elif priority == "MEDIUM":
        color = "orange"
    else:
        color = "blue"

    folium.Marker(
        [lat, lon],
        popup=f"Priority: {priority}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

st_folium(m, width=1100, height=600)

# -------------------------------------------------
# SYSTEM STATUS
# -------------------------------------------------
st.subheader("System Status")

if high_faults > 0:
    st.error("⚠ HIGH PRIORITY FAULTS DETECTED - Immediate Action Required")
elif total_faults > 0:
    st.warning("⚠ Faults Detected - Monitor System")
else:
    st.success("System Operating Normally")
