import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Premium Styling ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #0b1120);
    color: #f1f5f9;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
h1 {
    color: #60a5fa;
    text-align: center;
}
h2, h3 {
    color: #93c5fd;
}
.card {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-box {
    background: linear-gradient(90deg,#1e40af,#2563eb);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}

/* TEAM MODAL STYLING */
.team-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(6px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: fadeIn 0.3s ease-in-out;
}

.team-modal {
    position: relative;
    background: rgba(17,24,39,0.9);
    padding: 35px;
    border-radius: 18px;
    width: 420px;
    text-align: center;
    border: 2px solid transparent;
    background-clip: padding-box;
    box-shadow: 0 0 30px #3b82f6;
    animation: zoomIn 0.3s ease-in-out;
}

.team-modal h3 {
    margin-bottom: 10px;
    color: #60a5fa;
    font-size: 22px;
}

.team-modal p {
    font-size: 16px;
    color: #e5e7eb;
}

.team-close {
    position: absolute;
    top: 12px;
    right: 18px;
    font-size: 22px;
    cursor: pointer;
    color: white;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes zoomIn {
    from {transform: scale(0.8);}
    to {transform: scale(1);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Major Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Session State ----------------
if "selected_member" not in st.session_state:
    st.session_state.selected_member = None

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Home", "Analysis", "Insights", "About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")

members = {
    "Nilay Anand": "Worked on UI design and model integration.",
    "Mohit Paul": "Handled dataset preprocessing and feature selection.",
    "Ayush Raj": "Implemented K-Means and applied Elbow Method.",
    "Aditya Kumar": "Created visualizations and analyzed clustering results.",
    "Archita Rout": "Prepared documentation and explained methodology.",
    "Bhavya Rani": "Worked on business insights and presentation."
}

for name in members:
    if st.sidebar.button(name):
        st.session_state.selected_member = name

# ---------------- Premium Animated Popup ----------------
if st.session_state.selected_member is not None:

    name = st.session_state.selected_member
    role = members[name]

    st.markdown(f"""
    <div class="team-overlay" onclick="window.parent.postMessage({{type:'close'}}, '*')">
        <div class="team-modal" onclick="event.stopPropagation();">
            <span class="team-close" onclick="window.parent.postMessage({{type:'close'}}, '*')">✖</span>
            <h3>{name}</h3>
            <p>{role}</p>
        </div>
    </div>

    <script>
    window.addEventListener("message", (event) => {{
        if (event.data.type === "close") {{
            const overlay = document.querySelector(".team-overlay");
            if (overlay) overlay.remove();
        }}
    }});
    </script>
    """, unsafe_allow_html=True)

# ---------------- Load Dataset ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- Home ----------------
if menu == "Home":

    st.markdown("## Project Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm Used<br><b>K-Means Clustering</b></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This project applies K-Means clustering to segment mall customers based on income and spending behavior.
    The objective is to identify meaningful groups that support marketing strategy and business decision-making.
    </div>
    """, unsafe_allow_html=True)
