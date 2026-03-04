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

.stApp{
background:linear-gradient(135deg,#020617,#0f172a);
color:#e2e8f0;
}

.block-container{
border:1px solid rgba(96,165,250,0.25);
border-radius:16px;
padding:20px;
box-shadow:0px 0px 15px rgba(59,130,246,0.3);
}

section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#020617,#0f172a);
border-right:1px solid #1e293b;
}

h1{
color:#60a5fa;
text-align:center;
text-shadow:0px 0px 15px rgba(96,165,250,0.8);
}

h2{
color:#93c5fd;
}

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

.metric-box{
background:linear-gradient(135deg,#2563eb,#1d4ed8);
padding:16px;
border-radius:14px;
text-align:center;
box-shadow:0px 0px 20px rgba(59,130,246,0.6);
}

.stButton>button{
border-radius:14px;
padding:10px 18px;
border:none;
color:white;
background:linear-gradient(135deg,#06b6d4,#0891b2);
transition:all 0.3s ease;
box-shadow:0px 0px 10px rgba(6,182,212,0.6);
}

.stButton>button:hover{
transform:scale(1.05);
box-shadow:
0px 0px 8px rgba(6,182,212,0.9),
0px 0px 20px rgba(34,211,238,0.8),
0px 0px 30px rgba(103,232,249,0.7);
}

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

.dialog-footer{
margin-top:20px;
font-size:13px;
color:#94a3b8;
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
st.sidebar.markdown(
"""
<h2 style="
text-align:center;
color:#60a5fa;
text-shadow:0px 0px 12px #3b82f6;
">
Navigation
</h2>
""",
unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Select Section",
    ["🏠 Home", "📊 Analysis", "💡 Insights", "ℹ️ About"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
"""
<h3 style="color:#93c5fd;text-align:center;">👥 Project Team</h3>
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
        <div class="dialog-footer">
        Group 6 - ❤️ Thank You @CGU
        </div>
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
if menu == "🏠 Home":

    st.markdown("## Introduction")

    st.markdown("""
<div class="card">
Customer segmentation is used by businesses to understand customer behavior and group customers with similar spending patterns.
</div>
""", unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
elif menu == "📊 Analysis":

    if df is None:
        st.error("Mall_Customers.csv not found")
    else:

        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(include=['int64','float64']).columns

        features = st.multiselect(
            "Select Features",
            numeric_columns,
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
elif menu == "💡 Insights":

    st.markdown("""
<div class="card">
Customer segmentation helps businesses create targeted marketing strategies and identify premium customers.
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif menu == "ℹ️ About":

    st.markdown("""
<div class="card">
Mall Customer Segmentation using K-Means clustering developed as part of an AI Training Capstone Project at C V Raman Global University.
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
