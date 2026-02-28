import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Professional Styling ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
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
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}

.metric-box {
    background-color: #1e40af;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
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
st.sidebar.write("Nilay Anand")
st.sidebar.write("Mohit Paul")
st.sidebar.write("Ayush Raj")
st.sidebar.write("Aditya Kumar")
st.sidebar.write("Archita Rout")
st.sidebar.write("Bhavya Rani")

# ---------------- Load Dataset ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- Home Section ----------------
if menu == "Home":

    st.markdown("## Introduction")
    st.markdown("""
    <div class="card">
    Customer segmentation is a strategic marketing technique used to divide customers 
    into groups based on similar characteristics and purchasing behavior. 
    
    In this project, we implemented the K-Means clustering algorithm to segment 
    mall customers using Annual Income and Spending Score as primary features.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Problem Statement")
    st.markdown("""
    <div class="card">
    Businesses often face difficulty in identifying different customer categories 
    for targeted promotions. Without segmentation, marketing efforts may not 
    reach the appropriate audience effectively.
    
    This project aims to identify meaningful customer groups using 
    unsupervised machine learning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Methodology")
    st.markdown("""
    <div class="card">
    1. Data Loading and Preprocessing  
    2. Feature Selection (Income & Spending Score)  
    3. Determination of optimal clusters using Elbow Method  
    4. Implementation of K-Means Algorithm  
    5. Cluster Visualization  
    6. Model Evaluation using Silhouette Score  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Results and Observations")
    st.markdown("""
    <div class="card">
    The clustering algorithm successfully grouped customers into distinct 
    segments representing different spending behaviors.
    
    High-income high-spending customers were identified as premium customers, 
    while moderate and budget customers formed separate clusters.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Applications")
    st.markdown("""
    <div class="card">
    • Targeted marketing strategies  
    • Personalized offers  
    • Customer retention planning  
    • Strategic business decisions  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Future Scope")
    st.markdown("""
    <div class="card">
    • Inclusion of additional demographic features  
    • Use of advanced clustering techniques  
    • Real-time dashboard integration  
    • Enterprise-level deployment  
    </div>
    """, unsafe_allow_html=True)

# ---------------- Analysis Section ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found in project folder.")
    else:

        st.markdown("## Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Records<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm<br><b>K-Means</b></div>", unsafe_allow_html=True)

        st.markdown("### Dataset Preview")
        st.dataframe(df.head())

        st.markdown("### Feature Selection")

        features = st.multiselect(
            "Select Features",
            df.columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:

            X = df[features]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Elbow Method")

                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)

                fig1, ax1 = plt.subplots()
                ax1.plot(range(1, 11), wcss)
                ax1.set_xlabel("Clusters")
                ax1.set_ylabel("WCSS")
                st.pyplot(fig1)

            with col2:
                st.markdown("### Cluster Configuration")

                k = st.slider("Select Number of Clusters", 2, 10, 5)

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)

                df["Cluster"] = labels

                score = silhouette_score(X, labels)
                st.markdown(f"<div class='metric-box'>Silhouette Score<br><b>{round(score,2)}</b></div>", unsafe_allow_html=True)

            st.markdown("### Cluster Visualization")

            fig2, ax2 = plt.subplots()
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax2.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker="X",
                s=250
            )
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            st.pyplot(fig2)

            st.markdown("### Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

        else:
            st.warning("Please select at least two features.")

# ---------------- Insights ----------------
elif menu == "Insights":

    st.markdown("## Business Insights")

    st.markdown("""
    <div class="card">
    The segmentation model enables identification of customer types 
    such as premium customers, moderate spenders, and conservative buyers.
    
    These insights assist businesses in optimizing marketing campaigns 
    and improving customer engagement strategies.
    </div>
    """, unsafe_allow_html=True)

# ---------------- About ----------------
elif menu == "About":

    st.markdown("## About the Project")

    st.markdown("""
    <div class="card">
    Project Title: Mall Customer Segmentation using K-Means  
    Institution: C V Raman Global University  
    
    This project demonstrates the application of unsupervised machine learning 
    techniques in solving real-world business problems related to customer analytics.
    </div>
    """, unsafe_allow_html=True)
