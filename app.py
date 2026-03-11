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
st.set_page_config(page_title="Smart Grid Fault Detection", layout="wide")

# -------------------------------------------------
# CUSTOM STYLE
# -------------------------------------------------
st.markdown("""
<style>

.main {
    background-color:#0f172a;
}

h1,h2,h3 {
    color:#38bdf8;
}

.metric-card{
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
    font-weight:bold;
    font-size:18px;
}

.card1{background:#1e293b;}
.card2{background:#7c3aed;}
.card3{background:#ef4444;}
.card4{background:#f59e0b;}
.card5{background:#10b981;}

.stDataFrame {
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOGIN SYSTEM
# -------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

USER="admin"
PASS_HASH=hash_password("cloud123")

st.sidebar.title("🔐 Cloud Authentication")

username=st.sidebar.text_input("Username")
password=st.sidebar.text_input("Password",type="password")

if username!=USER or hash_password(password)!=PASS_HASH:
    st.title("⚡ Smart Grid Fault Detection System")
    st.warning("Login required to access dashboard")
    st.stop()

st.sidebar.success("Access Granted")

# -------------------------------------------------
# DATA GENERATION
# -------------------------------------------------
np.random.seed(42)

normal_voltage=np.random.normal(230,4,1200)
normal_current=np.random.normal(10,1.5,1200)

fault_voltage=np.random.normal(140,20,350)
fault_current=np.random.normal(28,8,350)

voltage=np.concatenate([normal_voltage,fault_voltage])
current=np.concatenate([normal_current,fault_current])

df=pd.DataFrame({
"Voltage":voltage,
"Current":current
})

# -------------------------------------------------
# VOLTAGE SEGMENTATION
# -------------------------------------------------
def segment_voltage(v):

    if v<200:
        return "LOW VOLTAGE"
    elif v>240:
        return "HIGH VOLTAGE"
    else:
        return "NORMAL"

df["Voltage_Segment"]=df["Voltage"].apply(segment_voltage)

# -------------------------------------------------
# MACHINE LEARNING MODEL
# -------------------------------------------------
model=IsolationForest(contamination=0.20,random_state=42)

df["Anomaly"]=model.fit_predict(df[["Voltage","Current"]])

df["Status"]=df["Anomaly"].map({
1:"NORMAL",
-1:"FAULT"
})

# -------------------------------------------------
# PRIORITY LOGIC
# -------------------------------------------------
def assign_priority(row):

    if row["Status"]=="FAULT":

        if row["Current"]>35:
            return "HIGH"

        elif row["Current"]>25:
            return "MEDIUM"

        else:
            return "LOW"

    return "NORMAL"

df["Priority"]=df.apply(assign_priority,axis=1)

# -------------------------------------------------
# DASHBOARD TITLE
# -------------------------------------------------
st.title("⚡ Tamil Nadu Smart Grid Fault Detection Dashboard")

st.divider()

# -------------------------------------------------
# METRICS
# -------------------------------------------------
total=len(df)
faults=len(df[df["Status"]=="FAULT"])
high=len(df[df["Priority"]=="HIGH"])
medium=len(df[df["Priority"]=="MEDIUM"])
low=len(df[df["Priority"]=="LOW"])

c1,c2,c3,c4,c5=st.columns(5)

with c1:
    st.markdown(f'<div class="metric-card card1">Total Data<br>{total}</div>',unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="metric-card card2">Total Faults<br>{faults}</div>',unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="metric-card card3">High Priority<br>{high}</div>',unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="metric-card card4">Medium Priority<br>{medium}</div>',unsafe_allow_html=True)

with c5:
    st.markdown(f'<div class="metric-card card5">Low Priority<br>{low}</div>',unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# CHARTS
# -------------------------------------------------
colA,colB=st.columns(2)

with colA:

    st.subheader("Fault Classification")

    fig,ax=plt.subplots(figsize=(6,5))

    sns.scatterplot(
    data=df,
    x="Voltage",
    y="Current",
    hue="Priority",
    palette={
    "HIGH":"red",
    "MEDIUM":"orange",
    "LOW":"yellow",
    "NORMAL":"green"
    })

    st.pyplot(fig)

with colB:

    st.subheader("Voltage Segmentation")

    seg=df["Voltage_Segment"].value_counts()

    st.bar_chart(seg)

st.divider()

# -------------------------------------------------
# GIS MAP
# -------------------------------------------------
st.subheader("Tamil Nadu Fault Map")

tn_lat=11.1271
tn_lon=78.6569

m=folium.Map(location=[tn_lat,tn_lon],zoom_start=7)

cluster=MarkerCluster().add_to(m)

fault_df=df[df["Status"]=="FAULT"]

for i in range(len(fault_df)):

    lat=np.random.uniform(8,13.5)
    lon=np.random.uniform(76,80.5)

    p=fault_df.iloc[i]["Priority"]

    if p=="HIGH":
        color="red"
    elif p=="MEDIUM":
        color="orange"
    else:
        color="blue"

    folium.Marker(
    [lat,lon],
    popup=f"Fault Priority: {p}",
    icon=folium.Icon(color=color)
    ).add_to(cluster)

st_folium(m,width=1100,height=550)

st.divider()

# -------------------------------------------------
# FAULT TABLE
# -------------------------------------------------
st.subheader("Fault Priority Table")

fault_table=df[df["Status"]=="FAULT"][[
"Voltage",
"Current",
"Voltage_Segment",
"Priority"
]]

st.dataframe(fault_table,use_container_width=True)

st.download_button(
"Download Fault CSV",
fault_table.to_csv(index=False),
"fault_data.csv"
)

# -------------------------------------------------
# SYSTEM STATUS
# -------------------------------------------------
st.subheader("System Status")

if high>0:
    st.error("⚠ High Priority Faults Detected")

elif faults>0:
    st.warning("⚠ Faults Present")

else:
    st.success("System Operating Normally")
