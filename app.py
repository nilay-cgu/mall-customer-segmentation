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
    1. Customer segmentation is a technique used to divide customers into groups.  
    2. It helps businesses understand customer behavior more effectively.  
    3. This project applies K-Means clustering for segmentation.  
    4. The main focus is on Annual Income and Spending Score.  
    5. The goal is to support better marketing decisions.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Problem Statement")

    st.markdown("""
    <div class="card">
    1. Businesses struggle to identify different types of customers.  
    2. Marketing campaigns often target all customers equally.  
    3. This reduces efficiency and increases cost.  
    4. The project aims to identify meaningful customer groups.  
    5. These groups help in designing targeted strategies.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Understanding")

    st.markdown("""
    <div class="card">
    1. The dataset contains 200 mall customers.  
    2. Features include Gender, Age, Annual Income, and Spending Score.  
    3. Income represents purchasing power.  
    4. Spending Score reflects buying behavior.  
    5. Income and Spending Score were selected for clustering.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Algorithm Explanation")

    st.markdown("""
    <div class="card">
    1. K-Means is an unsupervised machine learning algorithm.  
    2. It divides data into K clusters.  
    3. Each point is assigned to the nearest centroid.  
    4. Centroids are updated iteratively.  
    5. The process continues until clusters stabilize.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Model Evaluation")

    st.markdown("""
    <div class="card">
    1. The Elbow Method is used to determine optimal clusters.  
    2. It analyzes Within Cluster Sum of Squares (WCSS).  
    3. Silhouette Score measures cluster separation.  
    4. Higher Silhouette Score indicates better clustering.  
    5. Evaluation ensures reliability of results.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Applications")

    st.markdown("""
    <div class="card">
    1. Identifying premium customers.  
    2. Designing targeted marketing campaigns.  
    3. Improving customer retention strategies.  
    4. Supporting business decision-making.  
    5. Enhancing profitability through data insights.  
    </div>
    """, unsafe_allow_html=True)
