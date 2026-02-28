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
.card {
    background: rgba(30,41,59,0.6);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.metric-box {
    background: linear-gradient(90deg,#1e40af,#2563eb);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("---")

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

# ---------------- Native Dialog Popup ----------------
def show_member(name, role):
    @st.dialog(name)
    def dialog():
        st.write(role)
        st.divider()
        st.caption("Click outside the dialog or press Close to exit.")
        if st.button("Close"):
            st.rerun()
    dialog()

for name, role in members.items():
    if st.sidebar.button(name):
        show_member(name, role)

# ---------------- Load Dataset ----------------
try:
    df = pd.read_csv("Mall_Customers.csv")
except:
    df = None

# ---------------- Home ----------------
if menu == "Home":

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'>Total Customers<br><b>{df.shape[0]}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>Total Features<br><b>{df.shape[1]}</b></div>", unsafe_allow_html=True)
        col3.markdown("<div class='metric-box'>Algorithm<br><b>K-Means</b></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    This project applies K-Means clustering to segment mall customers 
    based on income and spending behavior.
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

            k = st.slider("Clusters", 2, 10, 5)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker="X",
                s=250
            )
            st.pyplot(fig)

# ---------------- Insights ----------------
elif menu == "Insights":

    st.markdown("""
    <div class="card">
    The clustering model provides meaningful insights into customer behavior 
    and helps businesses design targeted marketing strategies.
    </div>
    """, unsafe_allow_html=True)

# ---------------- About ----------------
elif menu == "About":

    st.markdown("""
    <div class="card">
    Mall Customer Segmentation using K-Means  
    C V Raman Global University  
    </div>
    """, unsafe_allow_html=True)
