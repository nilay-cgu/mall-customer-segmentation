import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("Mall Customer Segmentation")
st.write("- C V Raman Global University")

st.sidebar.title("Team Members")
st.sidebar.write("Nilay Anand")
st.sidebar.write("Mohit Paul")
st.sidebar.write("Ayush Raj")
st.sidebar.write("Aditya Kumar")
st.sidebar.write("Archita Rout")
st.sidebar.write("Bhavya Rani")

menu = st.sidebar.selectbox("Menu", ["Home", "Analysis", "About"])

if menu == "Home":
    st.subheader("Project Overview")
    st.write(
        "This project performs customer segmentation using K-Means clustering "
        "on the Mall Customers dataset."
    )

elif menu == "Analysis":
    st.subheader("Upload Dataset")
    file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        features = st.multiselect(
            "Select Features",
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
            ax1.plot(range(1, 11), wcss)
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
            ax2.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker="X",
                s=200
            )
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            st.pyplot(fig2)

            score = silhouette_score(X, labels)
            st.write("Silhouette Score:", round(score, 2))

            st.subheader("Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

        else:
            st.write("Please select at least two features.")

elif menu == "About":
    st.subheader("About Project")
    st.write(
        "This project applies K-Means clustering to segment customers "
        "based on income and spending behavior."
    )
