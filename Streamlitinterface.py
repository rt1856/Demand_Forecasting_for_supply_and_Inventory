import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dairy Forecast", layout="wide")
st.title("📈 Demand Forecasting Dashboard")

# Load data
df = pd.read_csv("forecasted_demand_next_30_days.csv")

if df.empty:
    st.error("No data loaded!")
else:
    product = st.selectbox("Select Product", df['Product'].unique())
    filtered = df[df['Product'] == product]

    st.subheader(f"Forecast for {product}")
    st.line_chart(filtered.set_index("Date")["Predicted_Demand"])
