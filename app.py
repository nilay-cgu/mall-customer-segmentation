import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Mall Customer Segmentation System",
    page_icon="🛍",
    layout="wide"
)

# -------------------------------
# Header Branding
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
    🛍 Mall Customer Segmentation System
    </h1>
    <h4 style='text-align: center;'>
    Major Project – C V RAMAN GLOBAL UNIVERSITY
    </h4>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Sidebar Branding
# -------------------------------
st.sidebar.title("👨‍💻 Project Team")

st.sidebar.markdown("""
### Team Members
- NILAY ANAND  
- MOHIT PAUL  
- ADITYA KUMAR  
- ARCHITA ROUT  
- BHAVYA RANI  

---

### Technology Used
- Python  
- Streamlit  
- Scikit-Learn  
- K-Means Clustering  
""")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Upload & Analysis", "About Project"]
)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.subheader("Project Overview")

    st.write("""
    This web-based application performs **Customer Segmentation**
    using the **K-Means Clustering Algorithm** on the Mall Customers dataset.

    ✔ Interactive User Interface  
    ✔ Elbow Method for Optimal Clusters  
    ✔ Silhouette Score Validation  
    ✔ Cluster Visualization  
    ✔ Download Clustered Dataset  
    """)

# -------------------------------
# ANALYSIS PAGE
# -------------------------------
elif page == "Upload & Analysis":

    st.subheader("Upload Mall_Customers.csv")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        features = st.multiselect(
            "Select Features for Clustering",
            df.columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:
            X = df[features]

            # -------------------------------
            # Elbow Method
            # -------------------------------
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
            ax1.set_title("Elbow Method Graph")
            st.pyplot(fig1)

            # -------------------------------
            # Select Cluster Number
            # -------------------------------
            k = st.slider("Select Number of Clusters", 2, 10, 5)

            # -------------------------------
            # Apply KMeans
            # -------------------------------
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            df["Cluster"] = labels

            # -------------------------------
            # Cluster Visualization
            # -------------------------------
            st.subheader("Cluster Visualization")

            fig2, ax2 = plt.subplots()
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax2.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker='X',
                s=250
            )
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            ax2.set_title("Customer Segments")
            st.pyplot(fig2)

            # -------------------------------
            # Silhouette Score
            # -------------------------------
            score = silhouette_score(X, labels)
            st.success(f"Silhouette Score: {round(score, 2)}")

            # -------------------------------
            # Cluster Statistics
            # -------------------------------
            st.subheader("Cluster Statistics")
            st.dataframe(df.groupby("Cluster").mean())

            # -------------------------------
            # Download Button
            # -------------------------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Clustered Dataset",
                csv,
                "clustered_customers.csv",
                "text/csv"
            )

        else:
            st.warning("Please select at least 2 features.")

# -------------------------------
# ABOUT PAGE
# -------------------------------
elif page == "About Project":
    st.subheader("About This Project")

    st.write("""
    **Project Title:** Mall Customer Segmentation Using K-Means  
    **Institution:** C V Raman Global University  
    **Domain:** Machine Learning & Data Analytics  

    This project segments customers into different groups 
    based on income and spending behavior to assist in 
    targeted marketing and business decision-making.
    """)

st.markdown("---")

st.markdown("""
<center>
<b>Developed By:</b> NILAY ANAND | MOHIT PAUL | ADITYA KUMAR | ARCHITA ROUT | BHAVYA RANI
</center>
""", unsafe_allow_html=True)
