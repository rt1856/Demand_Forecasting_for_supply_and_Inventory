# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("large_dairy_demand_forecast_sample.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- 1. Add Month
df['Month'] = df['Date'].dt.month

# --- 2. Add Season based on Indian months
def get_season(month):
    if month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Winter'

df['Season'] = df['Month'].apply(get_season)
df['Season_Code'] = df['Season'].map({'Summer': 0, 'Monsoon': 1, 'Winter': 2})

# --- 3. Add Festival Flag (based on known Indian festivals)
festivals = [
    '2025-01-14', '2025-03-29', '2025-08-15', '2025-10-02', 
    '2025-11-01', '2025-11-14', '2025-12-25'
]
df['Festival'] = df['Date'].isin(pd.to_datetime(festivals)).astype(int)

# --- 4. Simulate Political Index (scaled 0.4 to 1.0)
np.random.seed(42)
df['Political_Index'] = np.random.uniform(0.4, 1.0, len(df))

# --- 5. Simulate Export Trend per product/month (scaled 10 to 100)
df['Export_Trend'] = df.groupby(['Product', 'Month'])['Demand'].transform(lambda x: np.random.randint(10, 100))

# --- 6. Encode product as numeric
df['Product_Code'] = df['Product'].astype('category').cat.codes

# Preview updated DataFrame
print(df[['Date', 'Product', 'Demand', 'Temperature (°C)', 'Season', 'Season_Code', 'Festival',
          'Political_Index', 'Export_Trend']].head())

# Optional: Save updated dataset
df.to_csv("dairy_data_with_external_features.csv", index=False)
print("✅ External features added and saved to 'dairy_data_with_external_features.csv'")


# %%
