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
h2, h3 {
    color: #93c5fd;
}
.card {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-box {
    background: linear-gradient(90deg,#1e40af,#2563eb);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
}
.overlay-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.75);
    backdrop-filter: blur(6px);
    z-index: 999;
}
.popup-card {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(17,24,39,0.95);
    padding: 35px;
    border-radius: 18px;
    width: 420px;
    text-align: center;
    box-shadow: 0 0 35px #3b82f6;
    z-index: 1000;
    animation: fadeScale 0.3s ease-in-out;
}
@keyframes fadeScale {
    from {opacity:0; transform:translate(-50%, -60%) scale(0.9);}
    to {opacity:1; transform:translate(-50%, -50%) scale(1);}
}
.close-btn {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Major Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Session State ----------------
if "selected_member" not in st.session_state:
    st.session_state.selected_member = None

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

for name in members:
    if st.sidebar.button(name):
        st.session_state.selected_member = name

# ---------------- Popup ----------------
if st.session_state.selected_member is not None:

    name = st.session_state.selected_member
    role = members[name]

    # Dark Background
    st.markdown('<div class="overlay-bg"></div>', unsafe_allow_html=True)

    # Popup Card
    st.markdown(f"""
    <div class="popup-card">
        <h3 style="color:#60a5fa;">{name}</h3>
        <p style="color:#e5e7eb; font-size:16px;">{role}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,1,3])

    with col2:
        if st.button("Close"):
            st.session_state.selected_member = None
            st.rerun()

# ---------------- Load Dataset ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- Home ----------------
if menu == "Home":

    st.markdown("## Project Overview")

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm Used<br><b>K-Means Clustering</b></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This project applies K-Means clustering to segment mall customers 
    based on income and spending behavior. The objective is to identify 
    meaningful customer groups that support strategic marketing decisions.
    </div>
    """, unsafe_allow_html=True)

# ---------------- Analysis ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found.")
    else:

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
                wcss = []
                for i in range(1, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)

                fig1, ax1 = plt.subplots()
                ax1.plot(range(1, 11), wcss)
                st.pyplot(fig1)

            with col2:
                k = st.slider("Select Clusters", 2, 10, 5)
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                st.markdown(f"<div class='metric-box'>Silhouette Score<br><b>{round(score,2)}</b></div>", unsafe_allow_html=True)

            fig2, ax2 = plt.subplots()
            ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", s=250)
            st.pyplot(fig2)

# ---------------- Insights ----------------
elif menu == "Insights":

    st.markdown("""
    <div class="card">
    The clustering model provides meaningful insights into customer behavior. 
    High-value customers can be targeted with premium offers, while moderate 
    customers may respond to promotional strategies.
    </div>
    """, unsafe_allow_html=True)

# ---------------- About ----------------
elif menu == "About":

    st.markdown("""
    <div class="card">
    Project Title: Mall Customer Segmentation using K-Means  
    Institution: C V Raman Global University  
    
    This dashboard demonstrates practical implementation of 
    unsupervised machine learning in retail analytics.
    </div>
    """, unsafe_allow_html=True)
