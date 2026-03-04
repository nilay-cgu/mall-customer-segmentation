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

/* App background */
.stApp{
background:linear-gradient(135deg,#020617,#0f172a);
color:#e2e8f0;
}

/* AI glowing border */
.block-container{
border:1px solid rgba(96,165,250,0.25);
border-radius:16px;
padding:20px;
box-shadow:0px 0px 15px rgba(59,130,246,0.3);
}

/* Sidebar */
section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#020617,#0f172a);
border-right:1px solid #1e293b;
}

/* Title */
h1{
color:#60a5fa;
text-align:center;
text-shadow:0px 0px 15px rgba(96,165,250,0.8);
}

/* Cards */
.card{
background:rgba(30,41,59,0.55);
padding:22px;
border-radius:16px;
margin-bottom:16px;
border:1px solid rgba(255,255,255,0.05);
transition:all 0.3s ease;
}

.card:hover{
transform:translateY(-4px);
box-shadow:0px 0px 20px rgba(96,165,250,0.4);
}

/* Metric box */
.metric-box{
background:linear-gradient(135deg,#2563eb,#1d4ed8);
padding:16px;
border-radius:14px;
text-align:center;
box-shadow:0px 0px 20px rgba(59,130,246,0.6);
}

/* Neon Buttons */
.stButton>button{
border-radius:14px;
padding:10px 18px;
border:none;
color:white;
background:linear-gradient(135deg,#06b6d4,#0891b2);
transition:all 0.3s ease;
box-shadow:0px 0px 10px rgba(6,182,212,0.6);
}

/* Button hover glow */
.stButton>button:hover{
transform:scale(1.05);
box-shadow:
0px 0px 8px rgba(6,182,212,0.9),
0px 0px 20px rgba(34,211,238,0.8),
0px 0px 30px rgba(103,232,249,0.7);
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton>button{
background:rgba(30,41,59,0.7);
border:1px solid rgba(255,255,255,0.05);
}

/* Hover effect */
section[data-testid="stSidebar"] .stButton>button:hover{
background:#0891b2;
}

/* Floating profile icon */
.profile-icon{
width:90px;
height:90px;
border-radius:50%;
background:linear-gradient(135deg,#3b82f6,#2563eb);
display:flex;
align-items:center;
justify-content:center;
font-size:38px;
color:white;
margin:0 auto 15px auto;
animation:float 2s ease-in-out infinite;
box-shadow:0px 0px 20px #3b82f6;
}

@keyframes float{
0%{transform:translateY(0)}
50%{transform:translateY(-8px)}
100%{transform:translateY(0)}
}

/* Welcome animation */
@keyframes fadeIn{
0%{opacity:0;transform:translateY(-20px)}
100%{opacity:1;transform:translateY(0)}
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI Training Capstone Project - C V Raman Global University</h4>", unsafe_allow_html=True)

st.markdown("""
<div style="
height:2px;
background:linear-gradient(90deg,#2563eb,#06b6d4,#2563eb);
margin:25px 0;
box-shadow:0px 0px 10px #38bdf8;
"></div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Home", "Analysis", "Insights", "About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")

st.sidebar.markdown(
"""
<div style='font-size:12px;color:#60a5fa;margin-bottom:8px;'>
Click on a team member to view role
</div>
""",
unsafe_allow_html=True
)

members = {
"Nilay Anand":"Worked on UI design and model integration.",
"Mohit Paul":"Handled dataset preprocessing and feature selection.",
"Ayush Raj":"Implemented K-Means and applied Elbow Method.",
"Aditya Kumar":"Created visualizations and analyzed clustering results.",
"Archita Rout":"Prepared documentation and explained methodology.",
"Bhavya Rani":"Worked on business insights and presentation."
}

def show_member(name, role):
    @st.dialog(name)
    def dialog():
        st.markdown(f"""
        <div style="text-align:center;">
        <div class="profile-icon">{name[0]}</div>
        <h3>{name}</h3>
        <p>{role}</p>
        <p style="font-size:13px;color:#94a3b8;">
        Computer Science and Engineering : IoT and Cyber Security
        </p>
        </div>
        """, unsafe_allow_html=True)

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

# ---------------- HOME ----------------
if menu == "Home":

    st.markdown("""
<div style="
background:linear-gradient(135deg,#1e3a8a,#2563eb);
padding:35px;
border-radius:18px;
text-align:center;
margin-bottom:25px;
box-shadow:0px 0px 30px rgba(59,130,246,0.7);
animation:fadeIn 2s;
">

<h2 style="color:white;">
Mall Customer Segmentation using K-Means
</h2>

<p style="font-size:20px;color:#e0f2fe;">
AI Training Capstone Project using Machine Learning
</p>

</div>
""", unsafe_allow_html=True)

    st.markdown("### Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown("<div class='metric-box'><b>Algorithm</b><br>K-Means</div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-box'><b>Dataset</b><br>Mall Customers</div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-box'><b>Learning Type</b><br>Unsupervised</div>", unsafe_allow_html=True)

    st.markdown("## Introduction")

    st.markdown("""
<div class="card">
Customer segmentation helps businesses understand customer behavior
by grouping customers based on their spending patterns and income.
This project uses the K-Means clustering algorithm to identify
customer groups in a mall dataset.
</div>
""", unsafe_allow_html=True)

    st.markdown("## Problem Statement")

    st.markdown("""
<div class="card">
Businesses often treat all customers the same way.
However, different customers have different spending behaviors.
Customer segmentation helps businesses design targeted marketing strategies.
</div>
""", unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv not found")
    else:

        st.dataframe(df.head())

        features = st.multiselect(
            "Select Features",
            df.select_dtypes(include=['int64','float64']).columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:

            X = df[features]

            k = st.slider("Clusters",2,10,5)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            score = silhouette_score(X, labels)

            st.markdown(
                f"<div class='metric-box' style='width:250px;margin:auto;'>Silhouette Score<br><b>{round(score,2)}</b></div>",
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(X.iloc[:,0], X.iloc[:,1], c=labels)
            ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="X", s=200)
            st.pyplot(fig)

# ---------------- INSIGHTS ----------------
elif menu == "Insights":

    st.markdown("""
<div class="card">
High income and high spending customers represent premium customers.
Moderate customers respond well to promotional offers.
Customer segmentation helps businesses improve marketing efficiency.
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif menu == "About":

    st.markdown("""
<div class="card">
Mall Customer Segmentation using K-Means clustering.
Developed as an AI training capstone project at
C V Raman Global University.
</div>
""", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<hr>
<div style="text-align:center;color:#94a3b8;font-size:14px;">
Mall Customer Segmentation System <br>
AI Training Capstone Project <br>
Group 6 – C V Raman Global University
</div>
""", unsafe_allow_html=True)
