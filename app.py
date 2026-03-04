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
# ---------------- HOME ----------------
if menu == "🏠 Home":

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#1e3a8a,#2563eb);
    padding:35px;
    border-radius:18px;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0px 0px 30px rgba(59,130,246,0.7);
    animation:fadeIn 1.5s;
    ">
    <h2 style="color:white;">Mall Customer Segmentation using K-Means</h2>
    <p style="color:#e2e8f0;font-size:18px;">
    AI Training Capstone Project – Customer Behavior Analysis using Machine Learning
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-box">
    <b>Algorithm</b><br>
    K-Means Clustering
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-box">
    <b>Dataset</b><br>
    Mall Customer Dataset
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-box">
    <b>Type</b><br>
    Unsupervised Learning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    height:2px;
    background:linear-gradient(90deg,#2563eb,#06b6d4,#2563eb);
    margin:30px 0;
    box-shadow:0px 0px 10px #38bdf8;
    "></div>
    """, unsafe_allow_html=True)

    st.markdown("## Introduction")

    st.markdown("""
    <div class="card">
    Customer segmentation is an important technique used by businesses 
    to understand customer behavior. Instead of treating all customers 
    the same way, companies divide them into groups based on purchasing 
    patterns and financial characteristics.

    In this project, we use the K-Means clustering algorithm to segment 
    mall customers based on their Annual Income and Spending Score.
    This allows businesses to identify different types of customers 
    and design targeted marketing strategies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Problem Statement")

    st.markdown("""
    <div class="card">
    Businesses often struggle to understand customer behavior 
    because customers have different spending patterns.

    Without proper segmentation, companies may apply the same 
    marketing strategy to all customers, which reduces efficiency.

    This project solves this problem by applying clustering 
    techniques to identify meaningful customer groups.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Dataset Description")

    st.markdown("""
    <div class="card">
    The dataset contains mall customer information including:

    • Customer ID  
    • Gender  
    • Age  
    • Annual Income (k$)  
    • Spending Score (1–100)

    For clustering, we mainly focus on Annual Income and Spending Score 
    because these features strongly influence purchasing behavior.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Algorithm Used")

    st.markdown("""
    <div class="card">
    K-Means is an unsupervised machine learning algorithm used 
    to group similar data points into clusters.

    The algorithm works by:

    • Selecting a number of clusters (K)  
    • Assigning data points to the nearest centroid  
    • Updating centroids iteratively  
    • Repeating the process until clusters stabilize

    This technique helps identify natural groupings within the dataset.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🚀 How This System Can Be Improved")

    col1, col2 = st.columns(2)

    col1.markdown("""
    <div class="card">
    <h4>📊 More Features</h4>

    The current model uses only Annual Income and Spending Score 
    for clustering. The system can be improved by including more 
    features such as Age, Gender and purchase history.

    Using additional features will help the model create more 
    accurate customer groups and better business insights.
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="card">
    <h4>🤖 Advanced Algorithms</h4>

    Currently the system uses the K-Means clustering algorithm. 
    The model can be further improved by experimenting with 
    advanced clustering techniques such as:

    • Hierarchical Clustering  
    • DBSCAN  
    • Gaussian Mixture Models
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    col3.markdown("""
    <div class="card">
    <h4>📈 Real-Time Data Integration</h4>

    The system currently works with a static dataset. 
    Future versions could integrate real-time customer data 
    from retail systems or online platforms.

    This would allow businesses to perform live segmentation 
    and update marketing strategies dynamically.
    </div>
    """, unsafe_allow_html=True)

    col4.markdown("""
    <div class="card">
    <h4>🎯 Better Visualization</h4>

    Data visualization can be improved by adding interactive 
    charts and dashboards using tools like Plotly.

    Interactive dashboards help business managers 
    better understand customer patterns.
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
# ---------------- INSIGHTS ----------------
elif menu == "💡 Insights":

    st.markdown("## Customer Insights")

    st.markdown("""
    <div class="card">
    Customer segmentation helps businesses understand the purchasing 
    behavior of different types of customers. By analyzing income 
    and spending patterns, businesses can identify valuable customers 
    and design better marketing strategies.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.markdown("""
    <div class="card">
    <h4>💎 Premium Customers</h4>

    Customers with high income and high spending score 
    are considered premium customers. Businesses can 
    target them with premium products, loyalty programs 
    and exclusive offers.
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="card">
    <h4>🎯 Moderate Customers</h4>

    Customers with moderate income and spending behavior 
    respond well to promotional campaigns and discounts. 
    Targeted offers can increase their engagement and 
    improve sales performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    Businesses can use these insights to improve customer 
    satisfaction, optimize marketing strategies and 
    increase overall profitability.
    </div>
    """, unsafe_allow_html=True)


# ---------------- ABOUT ----------------
elif menu == "ℹ️ About":

    st.markdown("## About This Project")

    st.markdown("""
    <div class="card">
    <b>Project Title:</b> Mall Customer Segmentation using K-Means <br><br>

    This project demonstrates the use of machine learning 
    techniques to analyze customer behavior in retail environments.

    By applying the K-Means clustering algorithm, customers 
    are grouped based on their income and spending score, 
    allowing businesses to understand purchasing patterns 
    and develop targeted marketing strategies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Institution:</b> C V Raman Global University <br>
    <b>Branch:</b> Computer Science and Engineering – IoT & Cyber Security <br>
    <b>Group:</b> Group 6
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
