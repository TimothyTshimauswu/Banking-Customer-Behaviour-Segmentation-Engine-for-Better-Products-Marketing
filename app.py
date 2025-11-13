import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    with open("decision_tree_model.sav", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Put the feature names you trained on (all columns except 'Cluster')
FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

st.title("Customer Segmentation – Decision Tree App")

mode = st.radio("Select input mode", ["Single customer", "Upload CSV"])

# -------------------------
# Single customer input
# -------------------------
if mode == "Single customer":
    st.subheader("Enter customer details")

    inputs = {}
    for col in FEATURES:
        inputs[col] = st.number_input(col, value=0.0)

    if st.button("Predict cluster"):
        X_new = pd.DataFrame([inputs])
        pred = model.predict(X_new)[0]
        st.success(f"Predicted cluster: {int(pred)}")

# -------------------------
# Batch prediction via CSV
# -------------------------
else:
    st.subheader("Upload CSV for batch prediction")
    file = st.file_uploader("Upload a CSV file with customer features", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        # Ensure we only use the expected feature columns
        X_new = data[FEATURES]
        preds = model.predict(X_new)
        data["PredictedCluster"] = preds

        st.write("Preview with predicted clusters:")
        st.dataframe(data.head())

        csv_out = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_out,
            file_name="customers_with_clusters.csv",
            mime="text/csv",
        )