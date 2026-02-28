import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Styling ----------------
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
h2 {
    color: #93c5fd;
}
.card {
    background: rgba(30,41,59,0.6);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.metric-box {
    background: linear-gradient(90deg,#1e40af,#2563eb);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

/* Profile Animation */
.profile-icon {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    color: white;
    margin: 0 auto 15px auto;
    animation: float 2s ease-in-out infinite;
    box-shadow: 0 0 20px #3b82f6;
}
@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-8px);}
    100% {transform: translateY(0px);}
}
.dialog-footer {
    margin-top: 20px;
    font-size: 13px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Major Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

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

def show_member(name, role):
    @st.dialog(name)
    def dialog():
        initial = name[0]
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="profile-icon">{initial}</div>
            <h3>{name}</h3>
            <div style="font-size:13px; color:#94a3b8;">
                Computer Science and Engineering : IoT and Cyber Security
            </div>
            <hr style="margin:15px 0;">
            <p>{role}</p>
            <div class="dialog-footer">
                Group 6 - ❤️ Thank You @CGU
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Close"):
            st.rerun()

    dialog()

for name, role in members.items():
    if st.sidebar.button(name):
        show_member(name, role)

# ---------------- Load Dataset ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- HOME ----------------
if menu == "Home":

    st.markdown("## Introduction")
    st.markdown("""
    <div class="card">
    Customer segmentation is a data-driven technique used to divide customers into groups 
    based on similar characteristics. In this project, we applied the K-Means clustering 
    algorithm to segment mall customers using Annual Income and Spending Score.
    The objective is to identify patterns that help businesses make better marketing decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Description")
    st.markdown("""
    <div class="card">
    The dataset contains Customer ID, Gender, Age, Annual Income (k$), 
    and Spending Score (1–100). Income and Spending Score were selected 
    for clustering because they directly influence purchasing behavior.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## How K-Means Works")
    st.markdown("""
    <div class="card">
    K-Means is an unsupervised learning algorithm that divides data into K clusters.
    It assigns each data point to the nearest centroid and updates centroids iteratively 
    until cluster positions stabilize. The Elbow Method is used to determine the optimal K value.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Implementation Steps")
    st.markdown("""
    <div class="card">
    • Data preprocessing and feature selection  
    • Applying Elbow Method to find optimal clusters  
    • Implementing K-Means algorithm  
    • Evaluating results using Silhouette Score  
    • Visualizing clusters using scatter plots  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Applications")
    st.markdown("""
    <div class="card">
    • Targeted marketing campaigns  
    • Identifying premium customers  
    • Customer loyalty strategies  
    • Data-driven business decision making  
    </div>
    """, unsafe_allow_html=True)
