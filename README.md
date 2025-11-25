<p align="center">
  <img src="assets/Project_banner.png" width="100%" alt=" Banking Customer Behaviour Segmentation Engine for Better Products Marketing Banner"> 
</p>

<h1 align="center">🏦Banking Customer Behaviour Segmentation Engine for Better Products Marketing</h1>

<h3 align="center">
  PCA • K-Means Clustering • Decision Tree Classification • Streamlit Deployment
</h3>

<p align="center">
A full-stack behavioural analytics system I built to segment banking customers by spending patterns, income levels, and financial behaviour. The goal is to support targeted product design, improve marketing accuracy, and increase engagement and conversion.
</p>

## Executive Summary

This project started with my own experience as a South African banking customer. Standard Bank and Capitec frequently contacted me with product offers. Some matched my spending habits and income band. Others felt unrelated to my lifestyle. I wanted to understand how banks decide who receives credit cards, lifestyle bundles, investment products, or rewards programmes, and why some offers align while others do not.

I built a segmentation engine using engineered behavioural features, PCA, K-Means, and a Decision Tree to group customers into meaningful clusters. These clusters help banks target products more accurately, increase engagement, and reduce wasted marketing spend. I deployed a Streamlit interface so marketing, product, and BI teams can explore segments and identify high-value groups.

## Business Problem

Banks face four common challenges.

1. Low engagement from poorly targeted product offers.  
2. Marketing spend wasted on customers unlikely to convert.  
3. Limited insight into behaviour-driven product fit.  
4. Difficulty identifying cross-sell and up-sell opportunities.

This project answers:

- How customers differ in spending, income, and repayment behaviour.  
- Which behaviour patterns predict high or low engagement?  
- How segmentation improves targeting and campaign efficiency.  
- Which groups align with wealth, credit, lifestyle, or savings products?


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
## Technical Overview

### 1. Data Engineering

I engineered features such as:

- Income bands  
- Spending categories  
- Transaction frequency  
- Credit utilisation  
- Repayment behaviour  
- Lifestyle and activity indicators  

### 2. Feature Reduction and Clustering

I used PCA for dimensionality reduction and K-Means to create stable, interpretable segments.

The clusters show:

- Clear separation across behaviour groups  
- Low intra-cluster variance  
- High inter-cluster separation  

<p align="center">
  <img src="assets/PCA Plot.png" width="750"><br>
  <em>PCA projection showing separation between customer behaviour groups</em>
</p>


### 3. Segment Interpretation

I trained a Decision Tree to create simple rules that explain each segment. This helps business teams understand why customers fall into specific groups.
**Supervised Learning – Decision Tree Classifier**  
- Target: Cluster label assigned by K-Means  
- Train-test split: 70/30  
- Accuracy: ~92%  
- Metrics: Classification Report + Confusion Matrix  

### 4. Insights and Marketing Analytics

I built analytics summarising:

- Engagement potential  
- Spending and lifestyle patterns  
- Income and credit behaviour  
- Ideal product alignment per segment  

### 5. Deployment

I deployed a Streamlit interface that allows teams to explore segments, review customer behaviour summaries, and download insight reports.
- Models serialized with `joblib` and `pickle.`  
- Streamlit app runs locally or through Streamlit Cloud  
- PCA, scaler, and clustering pipelines integrated end-to-end  
- CSV batch scoring supported for business workflows  

🟢 Streamlit App Overview
The Streamlit application enables:

- 🧍 **Single-customer scoring**  
- 📤 **CSV upload for batch scoring**  
- 📈 **Real-time behavioural classification**  
- ✔️ **Prediction outputs for business users**
  
<p align="center">
  <img src="assets/streamlit_snapshot.png" width="750"><br>
  <em>Streamlit interface used by Banking Product and Marketing teams to segment a new customer and identify the best product fit</em>
</p>
This mirrors operational tools used by CRM, product, and credit teams in modern banks.

---
## Skills Demonstrated

Python  
Feature Engineering  
PCA  
K-Means  
Decision Trees  
Customer Analytics  
Streamlit  
Power BI  

## Results and Business Impact

This system improves marketing accuracy and operational efficiency.

1. Behaviour-aligned segmentation increases product relevance.  
2. Targeted campaigns deliver higher engagement.  
3. Marketing waste is reduced.  
4. Campaign planning becomes faster.  
5. Product and retention strategies improve through behavioural insight.

## How Teams Use This System

- Marketing teams use segments for targeted campaigns.  
- Product teams match offerings to the most relevant groups.  
- BI teams explore behaviour patterns across the customer base.  
- Management reviews segment-level revenue and engagement potential.

## Recommendations

1. Integrate segmentation into the CRM platform.  
2. Build personalised marketing journeys based on segment rules.  
3. Expand features using credit-bureau and richer transactional data.  
4. Add uplift modelling for predicted campaign response.

## Next Steps

1. Automate retraining using fresh behavioural data.  
2. Add drift monitoring for stability.  
3. Support multi-product segmentation.  
4. Deploy the engine on AWS for scale.
