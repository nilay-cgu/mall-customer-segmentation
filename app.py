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

/* Main background */
.stApp {
    background: linear-gradient(135deg,#020617,#0f172a);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#020617,#0f172a);
    border-right:1px solid #1e293b;
}

/* Title */
h1{
    color:#60a5fa;
    text-align:center;
    text-shadow:0px 0px 15px rgba(96,165,250,0.8);
}

/* Section titles */
h2{
    color:#93c5fd;
}

/* Cards */
.card{
    background: rgba(30,41,59,0.55);
    border-radius:16px;
    padding:22px;
    margin-bottom:16px;
    border:1px solid rgba(255,255,255,0.05);
    transition: all 0.3s ease;
}

/* Card hover */
.card:hover{
    transform: translateY(-4px);
    box-shadow:0px 0px 25px rgba(96,165,250,0.4);
}

/* Metric boxes */
.metric-box{
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    padding:16px;
    border-radius:14px;
    text-align:center;
    box-shadow:0px 0px 20px rgba(59,130,246,0.6);
}

/* Main buttons */
.stButton>button{
    border-radius:14px;
    padding:10px 18px;
    border:none;
    color:white;
    background: linear-gradient(135deg,#2563eb,#3b82f6);
    transition: all 0.3s ease;
    box-shadow:0px 0px 10px rgba(96,165,250,0.6);
}

/* Button hover glow */
.stButton>button:hover{
    transform: scale(1.05);
    box-shadow:
        0px 0px 8px rgba(59,130,246,0.9),
        0px 0px 20px rgba(96,165,250,0.8),
        0px 0px 30px rgba(147,197,253,0.7);
}

/* Sidebar team buttons */
section[data-testid="stSidebar"] .stButton>button{
    background: rgba(30,41,59,0.7);
    border:1px solid rgba(255,255,255,0.05);
}

/* Sidebar hover neon */
section[data-testid="stSidebar"] .stButton>button:hover{
    background:#2563eb;
    box-shadow:
        0px 0px 10px rgba(59,130,246,0.8),
        0px 0px 20px rgba(96,165,250,0.9);
}

/* Radio select section styling */
div[role="radiogroup"] > label{
    background: rgba(30,41,59,0.6);
    padding:8px 12px;
    border-radius:10px;
    margin-bottom:6px;
    transition:0.2s;
}

/* Radio hover */
div[role="radiogroup"] > label:hover{
    background:#1e40af;
    box-shadow:0px 0px 10px rgba(96,165,250,0.8);
}

/* Profile icon */
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
    animation: float 2s ease-in-out infinite;
    box-shadow:0px 0px 20px #3b82f6;
}

/* Floating animation */
@keyframes float{
    0%{transform:translateY(0px);}
    50%{transform:translateY(-8px);}
    100%{transform:translateY(0px);}
}

/* Footer text */
.dialog-footer{
    margin-top:20px;
    font-size:13px;
    color:#94a3b8;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI Training Capstone Project - C V Raman Global University</h4>", unsafe_allow_html=True)
st.markdown("---")

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
    <div style='font-size:12px; color:#60a5fa; margin-bottom:8px;'>
     Click on a team member to view role 
    </div>
    """,
    unsafe_allow_html=True
)

members = {
    "Nilay Anand": "Worked on UI design and model integration.",
    "Mohit Paul": "Handled dataset preprocessing and feature selection.",
    "Ayush Raj": "Implemented K-Means and applied Elbow Method.",
    "Aditya Kumar": "Created visualizations and analyzed clustering results.",
    "Archita Rout": "Prepared documentation and explained methodology.",
    "Bhavya Rani": "Worked on business insights and presentation."
}

def show_member(name, role):
    @st.dialog(name)
    def dialog():
        initial = name[0]
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="profile-icon">{initial}</div>
            <h3>{name}</h3>
            <div style="font-size:13px; color:#94a3b8;">
                Computer Science and Engineering : IoT and Cyber Security
            </div>
            <hr style="margin:15px 0;">
            <p>{role}</p>
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
if menu == "Home":

    st.markdown(
        """
        <div style="
        background:linear-gradient(135deg,#2563eb,#1e40af);
        padding:30px;
        border-radius:16px;
        text-align:center;
        margin-bottom:25px;
        box-shadow:0px 0px 25px rgba(96,165,250,0.6);
        ">
        <h2 style="color:white;">Mall Customer Segmentation using K-Means</h2>
        <p style="color:#e2e8f0;font-size:18px;">
        AI Training Capstone Project – Customer Behavior Analysis using Machine Learning
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        """
        <div class="metric-box">
        <b>Algorithm</b><br>
        K-Means Clustering
        </div>
        """,
        unsafe_allow_html=True
    )

    col2.markdown(
        """
        <div class="metric-box">
        <b>Dataset</b><br>
        Mall Customer Dataset
        </div>
        """,
        unsafe_allow_html=True
    )

    col3.markdown(
        """
        <div class="metric-box">
        <b>Type</b><br>
        Unsupervised Learning
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("## Introduction")

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Problem Statement")

    st.markdown(
        """
        <div class="card">
        Businesses often struggle to understand customer behavior 
        because customers have different spending patterns.

        Without proper segmentation, companies may apply the same 
        marketing strategy to all customers, which reduces efficiency.

        This project solves this problem by applying clustering 
        techniques to identify meaningful customer groups.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Dataset Description")

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Algorithm Used")

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )
    st.markdown("## 🚀 How This System Can Be Improved")

col1, col2 = st.columns(2)

col1.markdown(
"""
<div class="card">
<h4>📊 More Features</h4>

The current model uses only Annual Income and Spending Score 
for clustering. The system can be improved by including more 
features such as Age, Gender and purchase history.

Using additional features will help the model create more 
accurate customer groups and better business insights.
</div>
""",
unsafe_allow_html=True
)

col2.markdown(
"""
<div class="card">
<h4>🤖 Advanced Algorithms</h4>

Currently the system uses the K-Means clustering algorithm. 
The model can be further improved by experimenting with 
advanced clustering techniques such as:

• Hierarchical Clustering  
• DBSCAN  
• Gaussian Mixture Models

These algorithms may identify more complex patterns 
within the dataset.
</div>
""",
unsafe_allow_html=True
)

col3, col4 = st.columns(2)

col3.markdown(
"""
<div class="card">
<h4>📈 Real-Time Data Integration</h4>

The system currently works with a static dataset. 
Future versions could integrate real-time customer data 
from retail systems or online platforms.

This would allow businesses to perform live customer 
segmentation and adapt marketing strategies instantly.
</div>
""",
unsafe_allow_html=True
)

col4.markdown(
"""
<div class="card">
<h4>🎯 Better Visualization</h4>

Data visualization can be improved by adding interactive 
charts and dashboards using tools like Plotly or Power BI.

Interactive visualizations make it easier for managers 
to understand customer patterns and make strategic decisions.
</div>
""",
unsafe_allow_html=True
)

# ---------------- ANALYSIS ----------------
# ---------------- ANALYSIS ----------------
# ---------------- ANALYSIS ----------------
elif menu == "Analysis":

    if df is None:
        st.error("Mall_Customers.csv file not found.")
    else:

        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        features = st.multiselect(
            "Select Features (Only Numeric Columns Allowed)",
            numeric_columns,
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if len(features) >= 2:

            X = df[features]

            col1, col2 = st.columns([2,1])

            with col2:
                k = st.slider("Clusters", 2, 10, 5)

            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            score = silhouette_score(X, labels)

            # Smaller centered metric
            st.markdown(
                f"<div style='width:250px; margin:auto;' class='metric-box'>"
                f"Silhouette Score<br><b>{round(score,2)}</b></div>",
                unsafe_allow_html=True
            )

            # Smaller Graph
            fig, ax = plt.subplots(figsize=(6,4))  # 👈 size control here
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
            ax.scatter(
                kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                marker="X",
                s=200
            )
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])

            st.pyplot(fig, use_container_width=False)

        else:
            st.warning("Please select at least two numeric features.")

# ---------------- INSIGHTS ----------------
elif menu == "Insights":

    st.markdown("## Customer Insights")

    col1, col2 = st.columns(2)

    col1.markdown(
        """
        <div class="card">
        <h4>Premium Customers</h4>
        Customers with high income and high spending score 
        represent the most valuable segment for businesses.
        These customers are ideal targets for premium products.
        </div>
        """,
        unsafe_allow_html=True
    )

    col2.markdown(
        """
        <div class="card">
        <h4>Moderate Customers</h4>
        Customers with moderate income and spending patterns 
        respond well to promotional offers and discounts.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        Businesses can use customer segmentation to create 
        targeted marketing campaigns and improve customer 
        satisfaction while increasing profitability.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- ABOUT ----------------
elif menu == "About":

    st.markdown("## About This Project")

    st.markdown(
        """
        <div class="card">
        <b>Project Title:</b> Mall Customer Segmentation using K-Means <br><br>

        This project demonstrates the application of machine learning 
        techniques for analyzing customer behavior.

        By applying clustering algorithms, businesses can identify 
        patterns in customer spending habits and develop better 
        marketing strategies.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <b>Institution:</b> C V Raman Global University <br>
        <b>Branch:</b> Computer Science and Engineering – IoT & Cyber Security <br>
        <b>Group:</b> Group 6
        </div>
        """,
        unsafe_allow_html=True
    )
