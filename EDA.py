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
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("large_dairy_demand_forecast_sample.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 1. Demand trend for all products over time
plt.figure(figsize=(14, 6))
for product in df['Product'].unique():
    subset = df[df['Product'] == product]
    plt.plot(subset['Date'], subset['Demand'], label=product)

plt.title("Daily Demand Trend per Product")
plt.xlabel("Date")
plt.ylabel("Demand (Litres)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Boxplot: Demand distribution per product
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Product', y='Demand')
plt.title("Demand Distribution by Product")
plt.xticks(rotation=20)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Correlation heatmap (numeric features only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Average demand on holidays vs. non-holidays
holiday_avg = df.groupby("Holiday")["Demand"].mean()
print("\nAverage Demand on Holidays vs. Non-Holidays:\n", holiday_avg)

# 5. Inventory vs. Storage Utilization
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="Inventory", y="Storage_Util (%)", hue="Product")
plt.title("Inventory vs. Storage Utilization")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
