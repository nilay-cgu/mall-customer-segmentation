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
st.markdown("<h4 style='text-align:center;'>AI Training Capstone Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Home", "Analysis", "Insights", "About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")

st.sidebar.markdown(
    """
    <div style='font-size:12px; color:#60a5fa; margin-bottom:8px;'>
    👆 Click on a team member to view role details
    </div>
    """,
    unsafe_allow_html=True
)

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

    st.markdown("## Project Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm Used<br><b>K-Means Clustering</b></div>", unsafe_allow_html=True)

    st.markdown("## Introduction")
    st.markdown("""
    <div class="card">
    1. Customer segmentation divides customers into groups based on behavior.  
    2. It helps businesses understand purchasing patterns.  
    3. This project applies K-Means clustering for segmentation.  
    4. The focus is on Annual Income and Spending Score.  
    5. The objective is to improve marketing strategies.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Problem Statement")
    st.markdown("""
    <div class="card">
    1. Businesses often treat all customers equally.  
    2. This reduces marketing efficiency.  
    3. Identifying customer groups improves targeting.  
    4. Data-driven segmentation increases profitability.  
    5. The project solves this using clustering techniques.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Understanding")
    st.markdown("""
    <div class="card">
    1. Dataset contains 200 mall customers.  
    2. Features include Gender, Age, Income, and Spending Score.  
    3. Income represents purchasing power.  
    4. Spending Score indicates customer behavior.  
    5. Income and Spending Score were selected for clustering.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Algorithm Explanation")
    st.markdown("""
    <div class="card">
    1. K-Means is an unsupervised learning algorithm.  
    2. It divides data into K clusters.  
    3. Points are assigned to nearest centroid.  
    4. Centroids update iteratively.  
    5. The process stops when clusters stabilize.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Model Evaluation")
    st.markdown("""
    <div class="card">
    1. Elbow Method determines optimal clusters.  
    2. WCSS measures cluster compactness.  
    3. Silhouette Score measures cluster separation.  
    4. Higher score means better clustering.  
    5. Evaluation ensures model reliability.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Applications")
    st.markdown("""
    <div class="card">
    1. Identifying premium customers.  
    2. Targeted marketing campaigns.  
    3. Customer retention strategies.  
    4. Data-driven business decisions.  
    5. Improving overall profitability.  
    </div>
    """, unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
# ---------------- ANALYSIS ----------------
# ---------------- ANALYSIS ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found.")
    else:

        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        features = st.multiselect(
            "Select Features (Only Numeric Columns Allowed)",
            numeric_columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:

            X = df[features]

            col1, col2 = st.columns([2,1])

            with col2:
                k = st.slider("Clusters", 2, 10, 5)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            score = silhouette_score(X, labels)

            # Smaller centered metric
            st.markdown(
                f"<div style='width:250px; margin:auto;' class='metric-box'>"
                f"Silhouette Score<br><b>{round(score,2)}</b></div>",
                unsafe_allow_html=True
            )

            # Smaller Graph
            fig, ax = plt.subplots(figsize=(6,4))  # 👈 size control here
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker="X",
                s=200
            )
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])

            st.pyplot(fig, use_container_width=False)

        else:
            st.warning("Please select at least two numeric features.")

# ---------------- INSIGHTS ----------------
elif menu == "Insights":

    st.markdown("""
    <div class="card">
    1. High-income high-spending customers are premium customers.  
    2. Low-income low-spending customers require basic promotions.  
    3. Moderate customers can be targeted with personalized offers.  
    4. Segmentation improves marketing efficiency.  
    5. It supports strategic decision-making.  
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif menu == "About":

    st.markdown("""
    <div class="card">
    1. Project Title: Mall Customer Segmentation using K-Means.  
    2. Institution: C V Raman Global University.  
    3. Branch: Computer Science and Engineering (IoT and Cyber Security).  
    4. Developed as part of academic curriculum.  
    5. Focused on practical implementation of machine learning.  
    </div>
    """, unsafe_allow_html=True)
