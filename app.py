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

    st.markdown("## Project Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm Used<br><b>K-Means Clustering</b></div>", unsafe_allow_html=True)

    st.markdown("## Introduction")
    st.markdown("""
    <div class="card">
    Customer segmentation is a data analytics technique used to group customers 
    based on similar behavioral and financial characteristics. 
    In this project, we apply the K-Means clustering algorithm to classify mall customers 
    using Annual Income and Spending Score.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Description")
    st.markdown("""
    <div class="card">
    The dataset includes Customer ID, Gender, Age, Annual Income (k$), 
    and Spending Score (1–100). Income and Spending Score were selected 
    for clustering because they directly reflect purchasing behavior.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## How K-Means Works")
    st.markdown("""
    <div class="card">
    K-Means is an unsupervised learning algorithm that partitions data into K clusters.
    Each data point is assigned to the nearest centroid and centroids are updated 
    iteratively until convergence is achieved.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Implementation Steps")
    st.markdown("""
    <div class="card">
    • Data preprocessing and feature selection  
    • Determining optimal cluster count using Elbow Method  
    • Applying K-Means algorithm  
    • Evaluating clustering using Silhouette Score  
    • Visualizing results through scatter plots  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Advantages")
    st.markdown("""
    <div class="card">
    • Simple and computationally efficient  
    • Works well with structured numerical data  
    • Easy to interpret cluster results  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Applications")
    st.markdown("""
    <div class="card">
    • Targeted marketing campaigns  
    • Identifying high-value customers  
    • Customer retention strategies  
    • Business decision support  
    </div>
    """, unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
elif menu == "Analysis" and df is not None:

    st.dataframe(df.head())

    features = st.multiselect(
        "Select Features",
        df.columns,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(features) >= 2:
        X = df[features]

        k = st.slider("Clusters", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", s=250)
        st.pyplot(fig)

# ---------------- INSIGHTS ----------------
elif menu == "Insights":
    st.markdown("""
    <div class="card">
    The clustering model helps identify different types of customers.
    Businesses can use this segmentation to design focused marketing strategies.
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif menu == "About":
    st.markdown("""
    <div class="card">
    Mall Customer Segmentation using K-Means  
    C V Raman Global University  
    </div>
    """, unsafe_allow_html=True)
