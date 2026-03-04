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

.card{
background:rgba(30,41,59,0.55);
padding:22px;
border-radius:16px;
margin-bottom:16px;
border:1px solid rgba(255,255,255,0.05);
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
box-shadow:0px 0px 10px rgba(6,182,212,0.6);
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
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<h1>Mall Customer Segmentation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>AI Training Capstone Project - C V Raman Global University</h4>", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
menu = st.sidebar.radio(
"Navigation",
["🏠 Home","📊 Analysis","💡 Insights","ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Project Team")

members={
"Nilay Anand":"Worked on UI design and model integration.",
"Mohit Paul":"Handled dataset preprocessing.",
"Ayush Raj":"Implemented K-Means clustering.",
"Aditya Kumar":"Created visualizations.",
"Archita Rout":"Prepared documentation.",
"Bhavya Rani":"Worked on business insights."
}

for name,role in members.items():
    if st.sidebar.button(name):
        st.sidebar.info(role)

# ---------------- Dataset ----------------
try:
    df=pd.read_csv("Mall_Customers.csv")
except:
    df=None

# ---------------- HOME ----------------
if menu=="🏠 Home":

    st.markdown("### 📊 Project Overview")

    col1,col2,col3=st.columns(3)

    col1.markdown("""<div class="metric-box">
    <b>Algorithm</b><br>K-Means Clustering
    </div>""",unsafe_allow_html=True)

    col2.markdown("""<div class="metric-box">
    <b>Dataset</b><br>Mall Customer Dataset
    </div>""",unsafe_allow_html=True)

    col3.markdown("""<div class="metric-box">
    <b>Type</b><br>Unsupervised Learning
    </div>""",unsafe_allow_html=True)

    st.markdown("## Introduction")

    st.markdown("""<div class="card">
Customer segmentation is a technique used to divide customers into groups
based on their behavior and purchasing patterns.
In this project we use the K-Means clustering algorithm
to analyze mall customer data and identify different groups.
</div>""",unsafe_allow_html=True)

    st.markdown("## Problem Statement")

    st.markdown("""<div class="card">
Businesses often treat all customers the same.
However customers have different spending behaviors.
By identifying customer groups businesses can design
better marketing strategies and improve profits.
</div>""",unsafe_allow_html=True)

    st.markdown("## Dataset Description")

    st.markdown("""<div class="card">
Dataset includes:
• Customer ID  
• Gender  
• Age  
• Annual Income  
• Spending Score
</div>""",unsafe_allow_html=True)

    st.markdown("## Algorithm Used")

    st.markdown("""<div class="card">
K-Means clustering groups similar data points into clusters.
Steps:
1. Select number of clusters  
2. Assign points to nearest centroid  
3. Update centroids  
4. Repeat until clusters stabilize
</div>""",unsafe_allow_html=True)

    st.markdown("## 🚀 Future Improvements")

    col1,col2=st.columns(2)

    col1.markdown("""<div class="card">
<h4>More Features</h4>
Additional features like purchase history
and customer frequency could improve clustering.
</div>""",unsafe_allow_html=True)

    col2.markdown("""<div class="card">
<h4>Advanced Algorithms</h4>
Algorithms like DBSCAN or Hierarchical clustering
could provide better segmentation.
</div>""",unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
elif menu=="📊 Analysis":

    if df is None:
        st.error("Dataset not found")
    else:

        st.dataframe(df.head())

        features=st.multiselect(
        "Select Features",
        df.select_dtypes(include=['int64','float64']).columns,
        default=["Annual Income (k$)","Spending Score (1-100)"]
        )

        if len(features)>=2:

            X=df[features]

            k=st.slider("Clusters",2,10,5)

            model=KMeans(n_clusters=k)
            labels=model.fit_predict(X)

            score=silhouette_score(X,labels)

            st.markdown(f"""
            <div class="metric-box">
            Silhouette Score<br><b>{round(score,2)}</b>
            </div>
            """,unsafe_allow_html=True)

            fig,ax=plt.subplots(figsize=(6,4))
            ax.scatter(X.iloc[:,0],X.iloc[:,1],c=labels)
            ax.scatter(model.cluster_centers_[:,0],
                       model.cluster_centers_[:,1],
                       marker="X",s=200)

            st.pyplot(fig)

# ---------------- INSIGHTS ----------------
elif menu=="💡 Insights":

    st.markdown("## Customer Insights")

    col1,col2=st.columns(2)

    col1.markdown("""<div class="card">
<h4>Premium Customers</h4>
High income and high spending customers.
Businesses should target them with premium products.
</div>""",unsafe_allow_html=True)

    col2.markdown("""<div class="card">
<h4>Moderate Customers</h4>
Customers with moderate income.
Discount campaigns can increase engagement.
</div>""",unsafe_allow_html=True)

    st.markdown("""<div class="card">
Segmentation helps businesses improve marketing
and customer satisfaction.
</div>""",unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif menu=="ℹ️ About":

    st.markdown("## About This Project")

    st.markdown("""<div class="card">
Project: Mall Customer Segmentation using K-Means  
Institution: C V Raman Global University  
Branch: Computer Science and Engineering (IoT & Cyber Security)
</div>""",unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<hr>
<div style="text-align:center;color:#94a3b8;">
Mall Customer Segmentation System<br>
AI Training Capstone Project<br>
Group 6 – C V Raman Global University
</div>
""",unsafe_allow_html=True)
