<p align="center">
  <img src="assets/project_banner.png" width="100%" alt="Banking Customer Behaviour Segmentation Engine for Better Products Marketing Banner">
</p>

<h1 align="center">🏦Banking Customer Behaviour Segmentation Engine for Better Products Marketing</h1>

<h3 align="center">
  PCA • K-Means Clustering • Decision Tree Classification • Streamlit Deployment
</h3>

<p align="center">
  An end-to-end machine learning system for analysing credit-card customer behaviour,
  segmenting users, and classifying new customers for banking and financial services.
</p>


## Project Background

With the rise of AI and targeted marketing, banks are prioritising behaviour-driven analytics to improve product performance, strengthen customer engagement, and optimise credit decisions. Understanding how customers spend, repay, and use their credit facilities is essential for designing relevant savings products, personal loans, and wealth management offers. Retail banks such as FNB and ABSA, as well as major credit lenders, rely on this intelligence to run high-performing product and marketing campaigns. Industry research shows that behaviour-based segmentation can increase campaign conversion by **20–30%** and reduce early-stage delinquency by **10–15%**, making it a key driver of business value.

## 🧠 Project Overview

Banks rely heavily on behavioural analytics to improve product performance, credit strategies, and targeted marketing. With increasing competition from digital banks and fintechs, understanding customer activity—how they spend, repay, and use credit—is essential for portfolio growth.

This project delivers an end-to-end segmentation engine that enables:

- 🔍 Behavioural clustering of credit-card customers  
- 📊 PCA-powered dimensionality reduction  
- 🧠 Classification model for real-time customer scoring  
- 💼 Streamlit app for business-ready deployment  
- 📈 Insights for marketing, credit risk, and product decisioning  


## 🎯 Key Objectives

- Preprocess and scale customer behavioural data  
- Build PCA components for more interpretable features  
- Apply K-Means to generate meaningful customer segments  
- Train a Decision Tree classifier to predict segment labels  
- Deploy a Streamlit app for real-time scoring  
- Provide actionable insights for financial product teams  

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `Customer Data.csv` | Raw behavioural dataset |
| `clustered_customers_data.csv` | Final dataset with cluster labels |
| `kmeans_model.pkl` | Trained K-Means clustering model |
| `decision_tree_model.sav` | Saved classification model |
| `pca_model.pkl` | PCA transformer |
| `scaler.pkl` | StandardScaler used in pipeline |
| `app.py` | Streamlit application for deployment |
| `notebooks/` | EDA, clustering, PCA, and modelling |
| `assets/` | Visuals and project banner |

---

## 🧹 Data Preprocessing

- Imputed missing values in Credit_Limit and Minimum_Payments  
- Standardized all numerical features using StandardScaler  
- Removed Customer_ID and non-predictive fields  
- Engineered transaction frequency and utilisation features  
- Applied PCA to reduce multicollinearity across behavioural variables  

---

## 📈 Exploratory Data Analysis

- 📊 Distribution analysis for spending, payments, and balances  
- 🔥 Correlation heatmap to identify behavioural relationships  
- 🧭 Frequency vs. spending behaviour patterns  
- 💳 Cash advance trends and repayment behaviour  
- 🏷️ Identification of outlier customer groups  

---

## 🤖 Modelling Approach

### **Unsupervised Learning – K-Means Clustering**  
- Optimal K selected using the elbow method  
- Five behavioural segments identified  
- PCA plots used to visualize cluster separation  

### **Supervised Learning – Decision Tree Classifier**  
- Target: Cluster label assigned by K-Means  
- Train-test split: 70/30  
- Accuracy: ~92%  
- Metrics: Classification Report + Confusion Matrix  

---

## 🟢 Streamlit App Overview

The Streamlit application enables:

- 🧍 **Single-customer scoring**  
- 📤 **CSV upload for batch scoring**  
- 📈 **Real-time behavioural classification**  
- ✔️ **Prediction outputs for business users**

This mirrors operational tools used by CRM, product, and credit teams in modern banks.

---

## 🚀 Deployment

- Models serialized with `joblib` and `pickle`  
- Streamlit app runs locally or through Streamlit Cloud  
- PCA, scaler, and clustering pipelines integrated end-to-end  
- CSV batch scoring supported for business workflows  

---

## 🧠 Business Impact

This solution enables banking teams to:

- Identify high-value vs. low-engagement customers  
- Improve product targeting and personalised recommendations  
- Strengthen credit risk monitoring via behavioural indicators  
- Enhance portfolio profitability with segmentation insights  
- Automate customer profiling across marketing and credit teams  

Industry benchmarks show behaviour-driven segmentation can:

- Increase targeted campaign conversions by **20–30%**  
- Reduce early-stage delinquency by **10–15%**  
- Improve product uptake through personalised engagement  

This aligns directly with KPI-driven roles in analytics, CRM, and retail banking strategy.

---

## 🛠️ Tech Stack

- **Python:** Pandas, NumPy, Scikit-Learn  
- **Visualisation:** Matplotlib, Seaborn  
- **Dimensionality Reduction:** PCA  
- **Clustering:** K-Means  
- **Classification:** DecisionTreeClassifier  
- **Deployment:** Streamlit, pickle, joblib  

---

## 📌 Future Enhancements

- Add SHAP/LIME for behavioural explainability  
- Integrate real-time transactional feeds  
- Expand segmentation using HDBSCAN or GMM  
- Add customer lifetime value (CLV) scoring  
- Build a Power BI dashboard for business teams  

---
