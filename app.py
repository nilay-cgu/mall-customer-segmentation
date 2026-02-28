
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍 Mall Customer Segmentation using K-Means")

uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Features for Clustering")
    features = st.multiselect(
        "Choose features",
        df.columns,
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    if len(features) >= 2:
        X = df[features]

        st.subheader("Elbow Method")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

        k = st.slider("Select Number of Clusters", 2, 10, 5)

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        df["Cluster"] = labels

        st.subheader("Cluster Visualization")
        fig2, ax2 = plt.subplots()
        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
        ax2.scatter(kmeans.cluster_centers_[:, 0],
                    kmeans.cluster_centers_[:, 1],
                    marker='X')
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])
        st.pyplot(fig2)

        score = silhouette_score(X, labels)
        st.subheader(f"Silhouette Score: {round(score, 2)}")

        st.subheader("Cluster Statistics")
        st.dataframe(df.groupby("Cluster").mean())

    else:
        st.warning("Please select at least 2 features.")
