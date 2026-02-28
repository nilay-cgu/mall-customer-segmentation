import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Custom Light Styling ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f9;
}
h1 {
    color: #1e3a8a;
}
h2, h3 {
    color: #2563eb;
}
section[data-testid="stSidebar"] {
    background-color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center;'>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
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

# ---------------- Load Dataset Automatically ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- Home ----------------
if menu == "Home":

    st.markdown("## Project Overview")

    st.write("""
    This project performs customer segmentation using the K-Means 
    clustering algorithm on the Mall Customers dataset. 
    The objective is to identify meaningful customer groups 
    based on annual income and spending behavior.
    """)

    st.markdown("### Dataset Information")

    if df is not None:
        st.write("Total Records:", df.shape[0])
        st.write("Total Features:", df.shape[1])
    else:
        st.warning("Mall_Customers.csv file not found in project folder.")

    st.markdown("### Implementation Steps")

    st.write("""
    • Data preprocessing  
    • Feature selection  
    • Elbow method analysis  
    • K-Means clustering  
    • Cluster evaluation using Silhouette Score  
    """)

# ---------------- Analysis ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found. Please keep the file in the project folder.")
    else:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:

            X = df[features]

            col1, col2 = st.columns(2)

            # Elbow Method
            with col1:
                st.markdown("### Elbow Method")

                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)

                fig1, ax1 = plt.subplots()
                ax1.plot(range(1, 11), wcss)
                ax1.set_xlabel("Number of Clusters")
                ax1.set_ylabel("WCSS")
                ax1.set_title("Elbow Graph")
                st.pyplot(fig1)

            # Cluster Configuration
            with col2:
                st.markdown("### Cluster Configuration")

                k = st.slider("Select Number of Clusters", 2, 10, 5)

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)

                df["Cluster"] = labels

                score = silhouette_score(X, labels)
                st.metric("Silhouette Score", round(score, 2))

            # Cluster Visualization
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
            ax2.set_title("Customer Segments")
            st.pyplot(fig2)

            st.markdown("### Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

        else:
            st.warning("Please select at least two features.")

# ---------------- Insights ----------------
elif menu == "Insights":

    st.subheader("Business Insights")

    st.write("""
    The clustering results help identify different types of customers 
    such as high-income high-spending customers and conservative buyers. 
    These insights assist businesses in targeted marketing 
    and strategic planning.
    """)

# ---------------- About ----------------
elif menu == "About":

    st.subheader("About the Project")

    st.write("""
    Project Title: Mall Customer Segmentation using K-Means  
    Institution: C V Raman Global University  

    This project demonstrates practical implementation of 
    unsupervised machine learning techniques for real-world 
    customer behavior analysis.
    """)
