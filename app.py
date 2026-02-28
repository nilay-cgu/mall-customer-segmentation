import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# -------- Dark Theme Styling --------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stApp {
    background-color: #0E1117;
}
h1, h2, h3, h4 {
    color: #4DA6FF;
}
</style>
""", unsafe_allow_html=True)

# -------- Header --------
st.title("Mall Customer Segmentation")
st.write("Major Project - C V Raman Global University")

st.markdown("---")

# -------- Sidebar --------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["Home", "Customer Analysis", "About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")
st.sidebar.write("Nilay Anand")
st.sidebar.write("Mohit Paul")
st.sidebar.write("Aditya Kumar")
st.sidebar.write("Archita Rout")
st.sidebar.write("Bhavya Rani")

# -------- Home --------
if menu == "Home":
    st.subheader("Project Overview")
    st.write(
        "This application performs customer segmentation using "
        "K-Means clustering on the Mall Customers dataset. "
        "It helps identify customer groups based on income and spending patterns."
    )

# -------- Analysis --------
elif menu == "Customer Analysis":
    st.subheader("Upload Dataset")
    file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:
            X = df[features]

            # Elbow Method
            st.subheader("Elbow Method")

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

            # Cluster Selection
            k = st.slider("Select Number of Clusters", 2, 10, 5)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            df["Cluster"] = labels

            # Cluster Plot
            st.subheader("Cluster Visualization")

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

            # Silhouette Score
            score = silhouette_score(X, labels)
            st.write("Silhouette Score:", round(score, 2))

            # Cluster Stats
            st.subheader("Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

        else:
            st.warning("Please select at least two features.")

# -------- About --------
elif menu == "About":
    st.subheader("About the Project")
    st.write(
        "This project applies K-Means clustering to segment customers "
        "based on income and spending behavior. "
        "It is developed as a major project for C V Raman Global University."
    )
