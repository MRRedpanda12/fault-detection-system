
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import folium
from streamlit_folium import folium_static

st.title("Cloud-Based Fault Detection System")

# Generate Data
np.random.seed(42)

normal_voltage = np.random.normal(230, 3, 200)
normal_current = np.random.normal(10, 1, 200)

fault_voltage = np.random.normal(150, 10, 40)
fault_current = np.random.normal(25, 5, 40)

voltage = np.concatenate([normal_voltage, fault_voltage])
current = np.concatenate([normal_current, fault_current])

df = pd.DataFrame({
    "Voltage": voltage,
    "Current": current
})

# ML Model
model = IsolationForest(contamination=0.15, random_state=42)
df["Anomaly"] = model.fit_predict(df)
df["Status"] = df["Anomaly"].map({1: "NORMAL", -1: "FAULT"})

st.subheader("Fault Detection Output")
st.write(df.head())

# Visualization
st.subheader("Fault Visualization")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="Voltage",
    y="Current",
    hue="Status",
    palette={"NORMAL": "green", "FAULT": "red"},
    ax=ax
)
st.pyplot(fig)

# Digital Twin Simulation
st.subheader("Digital Twin Simulation")

source_voltage = 230
line_impedance = {
    "Segment_A": 0.05,
    "Segment_B": 0.08,
    "Segment_C": 0.10
}

for seg in line_impedance:
    fault_current = source_voltage / (line_impedance[seg] + 0.5)
    st.write(f"{seg} Fault Current: {round(fault_current,2)} A")

# GIS Map
st.subheader("GIS Fault Mapping")

m = folium.Map(location=[11.0168, 76.9558], zoom_start=13)

fault_locations = [
    (11.0168, 76.9558),
    (11.0250, 76.9600),
    (11.0100, 76.9400)
]

for lat, lon in fault_locations:
    folium.Marker(
        [lat, lon],
        popup="Fault Detected",
        icon=folium.Icon(color="red")
    ).add_to(m)

folium_static(m)

# Summary
st.subheader("System Summary")
st.write("Total Data Points:", len(df))
st.write("Detected Faults:", len(df[df["Status"] == "FAULT"]))
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import folium
from streamlit_folium import folium_static

st.set_page_config(layout="wide")
st.title("Cloud-Based Fault Detection Using Digital Twin & GIS")

# -------------------------------------------------
#  Generate Simulated Feeder Data
# -------------------------------------------------
np.random.seed(42)

normal_voltage = np.random.normal(230, 3, 200)
normal_current = np.random.normal(10, 1, 200)

fault_voltage = np.random.normal(150, 10, 40)
fault_current = np.random.normal(25, 5, 40)

voltage = np.concatenate([normal_voltage, fault_voltage])
current = np.concatenate([normal_current, fault_current])

df = pd.DataFrame({
    "Voltage": voltage,
    "Current": current
})

# -------------------------------------------------
#  ML Fault Detection
# -------------------------------------------------
model = IsolationForest(contamination=0.15, random_state=42)
df["Anomaly"] = model.fit_predict(df)
df["Status"] = df["Anomaly"].map({1: "NORMAL", -1: "FAULT"})

total_points = len(df)
total_faults = len(df[df["Status"] == "FAULT"])

# -------------------------------------------------
#  System Summary
# -------------------------------------------------
st.subheader("System Summary")
col1, col2 = st.columns(2)
col1.metric("Total Data Points", total_points)
col2.metric("Detected Faults", total_faults)

# -------------------------------------------------
#  Display Full Dataset
# -------------------------------------------------
st.subheader("Complete Dataset (All Normal + Fault Points)")
st.dataframe(df)

st.subheader("Normal Data Points")
st.dataframe(df[df["Status"] == "NORMAL"])

st.subheader("Fault Data Points")
st.dataframe(df[df["Status"] == "FAULT"])

# -------------------------------------------------
#  Visualization
# -------------------------------------------------
st.subheader("Fault Visualization")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Voltage",
    y="Current",
    hue="Status",
    palette={"NORMAL": "green", "FAULT": "red"},
    ax=ax
)
plt.title("ML-Based Fault Detection")
st.pyplot(fig)

# -------------------------------------------------
#  Digital Twin Simulation
# -------------------------------------------------
st.subheader("Digital Twin Fault Current Calculation")

source_voltage = 230
line_impedance = {
    "Segment_A": 0.05,
    "Segment_B": 0.08,
    "Segment_C": 0.10
}

for seg in line_impedance:
    fault_current = source_voltage / (line_impedance[seg] + 0.5)
    st.write(f"{seg} Fault Current: {round(fault_current,2)} A")

# -------------------------------------------------
#  GIS Fault Mapping (Dynamic Based on ML Output)
# -------------------------------------------------
st.subheader("GIS Fault Mapping")

center_lat = 11.0168
center_lon = 76.9558

m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Add same number of markers as detected faults
for i in range(total_faults):
    lat = center_lat + np.random.uniform(-0.02, 0.02)
    lon = center_lon + np.random.uniform(-0.02, 0.02)

    folium.Marker(
        [lat, lon],
        popup="Detected Fault Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

folium_static(m)

# -------------------------------------------------
#  Final System Conclusion
# -------------------------------------------------
st.subheader("System Conclusion")

if total_faults > 0:
    st.error("Fault Condition Detected in Feeder Network")
else:
    st.success("System Operating Normally")
