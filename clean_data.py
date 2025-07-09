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

# Load data
df = pd.read_csv("large_dairy_demand_forecast_sample.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values(by='Date')

# Check for nulls
print(df.isnull().sum())

# Drop duplicates if any
df = df.drop_duplicates()

# Feature engineering
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Optional: Rolling average and lag demand
df['Demand_Rolling7'] = df.groupby('Product')['Demand'].transform(lambda x: x.rolling(7).mean())
df['Lag_1'] = df.groupby('Product')['Demand'].shift(1)

# Drop rows with NaN created due to rolling/lag
df.dropna(inplace=True)

# Save cleaned version (optional)
df.to_csv("cleaned_dairy_data.csv", index=False)


# %%
