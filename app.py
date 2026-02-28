import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- Page Config ----------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# ---------------- Custom Dark Styling ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header Section ----------------
st.markdown("<h1 style='text-align:center;'>Mall Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Major Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.title("Project Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Home", "Data Analysis", "About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")
st.sidebar.write("Nilay Anand")
st.sidebar.write("Mohit Paul")
st.sidebar.write("Aditya Kumar")
st.sidebar.write("Archita Rout")
st.sidebar.write("Bhavya Rani")

# ---------------- Home ----------------
if menu == "Home":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Overview")
        st.write("""
        This dashboard performs customer segmentation using
        K-Means clustering on the Mall Customers dataset.
        The objective is to identify distinct customer groups
        based on income and spending patterns.
        """)

    with col2:
        st.subheader("Key Features")
        st.write("""
        • Interactive dataset upload  
        • Elbow method visualization  
        • Dynamic cluster selection  
        • Silhouette score evaluation  
        • Cluster statistics summary  
        """)

# ---------------- Data Analysis ----------------
elif menu == "Data Analysis":

    st.subheader("Upload Mall_Customers.csv")
    file = st.file_uploader("Upload Dataset", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.markdown("### Dataset Preview")
        st.dataframe(df.head())

        st.markdown("### Feature Selection")
        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:
            X = df[features]

            col1, col2 = st.columns(2)

            # -------- Elbow Method --------
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
                ax1.set_title("Elbow Graph")
                st.pyplot(fig1)

            # -------- Cluster Selection --------
            with col2:
                st.markdown("### Cluster Configuration")
                k = st.slider("Select Number of Clusters", 2, 10, 5)

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)

                df["Cluster"] = labels

                score = silhouette_score(X, labels)
                st.metric("Silhouette Score", round(score, 2))

            # -------- Cluster Visualization --------
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

            # -------- Statistics --------
            st.markdown("### Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

        else:
            st.warning("Please select at least two features.")

# ---------------- About ----------------
elif menu == "About":
    st.subheader("About the Project")
    st.write("""
    Project Title: Mall Customer Segmentation using K-Means  
    Institution: C V Raman Global University  

    This project applies unsupervised learning to identify 
    distinct customer segments for better marketing strategy planning.
    """)
