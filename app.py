import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Configuration ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Styling ----------------
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

    st.markdown("## Project Overview")

    col1, col2, col3 = st.columns(3)

    if df is not None:
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm Used<br><b>K-Means Clustering</b></div>", unsafe_allow_html=True)

    st.markdown("## Introduction")
    st.markdown("""
    <div class="card">
    Customer segmentation is a data analytics technique used to group customers 
    based on similar behavioral and financial characteristics. 
    This project applies K-Means clustering to classify mall customers 
    using Annual Income and Spending Score as primary attributes.
    The objective is to identify patterns that support strategic marketing decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Description")
    st.markdown("""
    <div class="card">
    The dataset consists of mall customer information including Customer ID, 
    Gender, Age, Annual Income (k$), and Spending Score (1–100).
    For clustering, income and spending score were selected 
    because they directly influence purchasing behavior.
    The dataset provides a practical example of real-world retail analytics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## How K-Means Works")
    st.markdown("""
    <div class="card">
    K-Means is an unsupervised machine learning algorithm 
    that partitions data into K distinct clusters.
    It assigns data points to clusters by minimizing the 
    Within Cluster Sum of Squares (WCSS).
    The algorithm iteratively updates cluster centroids 
    until convergence is achieved.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Implementation Steps")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card">
        • Data preprocessing  
        • Feature selection  
        • Elbow method analysis  
        • Optimal cluster determination  
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card">
        • Model training using K-Means  
        • Cluster visualization  
        • Silhouette score evaluation  
        • Interpretation of results  
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Advantages")
    st.markdown("""
    <div class="card">
    • Simple and computationally efficient  
    • Works well with large datasets  
    • Easy to interpret results  
    • Effective for spherical cluster structures  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Limitations")
    st.markdown("""
    <div class="card">
    • Requires predefined number of clusters  
    • Sensitive to initial centroid selection  
    • Not suitable for non-spherical clusters  
    • Affected by outliers  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Real-World Applications")
    st.markdown("""
    <div class="card">
    • Targeted marketing campaigns  
    • Customer loyalty segmentation  
    • Sales strategy optimization  
    • Business intelligence analytics  
    </div>
    """, unsafe_allow_html=True)

# ---------------- Analysis ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found in project folder.")
    else:

        st.markdown("## Dataset Preview")
        st.dataframe(df.head())

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
                st.markdown("### Cluster Selection")

                k = st.slider("Select Number of Clusters", 2, 10, 5)

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)

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

# ---------------- Insights ----------------
elif menu == "Insights":

    st.markdown("""
    <div class="card">
    The clustering model provides actionable insights into customer purchasing behavior. 
    High-income high-spending customers can be targeted with premium offers, 
    while moderate spenders may respond to promotional campaigns. 
    These insights enable businesses to allocate marketing resources efficiently 
    and improve overall profitability.
    </div>
    """, unsafe_allow_html=True)

# ---------------- About ----------------
elif menu == "About":

    st.markdown("""
    <div class="card">
    Project Title: Mall Customer Segmentation using K-Means  
    Institution: C V Raman Global University  
    
    This project demonstrates practical implementation of 
    unsupervised learning techniques in retail analytics. 
    It integrates data preprocessing, clustering, evaluation, 
    and visualization into an interactive dashboard system.
    </div>
    """, unsafe_allow_html=True)
