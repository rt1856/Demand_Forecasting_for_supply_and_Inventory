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
pip install XGBoost

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

# Preview the new DataFrame
df[['Date', 'Product', 'Demand', 'Temperature (°C)', 'Season', 'Festival', 'Political_Index', 'Export_Trend']].head()

# Save updated dataset
df.to_csv("dairy_data_with_external_features.csv", index=False)
print("✅ External features added and saved to 'dairy_data_with_external_features.csv'")


# %%
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Encode product
df['Product_Code'] = df['Product'].astype('category').cat.codes

# Feature columns
features = ['Product_Code', 'Temperature (°C)', 'Season_Code', 'Holiday', 'Festival',
            'Political_Index', 'Export_Trend']
target = 'Demand'

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost Forecast - MAE: {mae:.2f}, RMSE: {rmse:.2f}")


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(y_test.values[:30], label='Actual', marker='o')
plt.plot(y_pred[:30], label='Predicted', marker='x')
plt.title("Actual vs Predicted Demand (First 30 Samples)")
plt.legend()
plt.grid(True)
plt.show()


# %% [markdown]
# Stimulate Future Data

# %%

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Start from day after last date in original dataset
last_date = pd.to_datetime(df['Date']).max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

# Products list
products = df['Product'].unique()

# Create future DataFrame
future_data = []

for date in future_dates:
    for product in products:
        temp = np.random.randint(28, 38)  # Simulate temperature
        month = date.month
        season_code = 0 if month in [3, 4, 5] else 1 if month in [6, 7, 8, 9] else 2
        holiday = 1 if date.weekday() == 6 else 0
        festival = 1 if str(date.date()) in ['2025-08-15', '2025-10-02', '2025-11-01', '2025-12-25'] else 0
        political_index = np.random.uniform(0.4, 1.0)
        export_trend = np.random.randint(10, 100)
        product_code = df[df['Product'] == product]['Product_Code'].iloc[0]
        
        future_data.append([date, product, product_code, temp, season_code, holiday,
                            festival, political_index, export_trend])

# Create DataFrame
future_df = pd.DataFrame(future_data, columns=[
    'Date', 'Product', 'Product_Code', 'Temperature (°C)', 'Season_Code',
    'Holiday', 'Festival', 'Political_Index', 'Export_Trend'
])

# Preview
future_df.head()


# %% [markdown]
# Predict Demand for Future Data

# %%
# Prepare features for model
X_future = future_df[['Product_Code', 'Temperature (°C)', 'Season_Code', 'Holiday',
                      'Festival', 'Political_Index', 'Export_Trend']]

# Predict
future_df['Predicted_Demand'] = model.predict(X_future)

# Preview result
print(future_df[['Date', 'Product', 'Predicted_Demand']].head(10))

# Optional: Save to CSV
future_df.to_csv("forecasted_demand_next_30_days.csv", index=False)
print("✅ Future demand forecast saved as 'forecasted_demand_next_30_days.csv'")


# %% [markdown]
# Plot of Every Product

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Load forecast data
future_df = pd.read_csv("forecasted_demand_next_30_days.csv")
future_df['Date'] = pd.to_datetime(future_df['Date'])

# Get list of products
products = future_df['Product'].unique()

# Plot each product's forecast
for product in products:
    product_df = future_df[future_df['Product'] == product]

    plt.figure(figsize=(10, 4))
    plt.plot(product_df['Date'], product_df['Predicted_Demand'], marker='o', color='teal')
    plt.title(f"📈 Forecasted Demand for {product} - Next 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Predicted Demand (Litres)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# Forecast For all Combined

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Load the forecasted data (if not already in memory)
future_df = pd.read_csv("forecasted_demand_next_30_days.csv")
future_df['Date'] = pd.to_datetime(future_df['Date'])

# Set up the plot
plt.figure(figsize=(14, 6))

# Plot each product's forecast
for product in future_df['Product'].unique():
    subset = future_df[future_df['Product'] == product]
    plt.plot(subset['Date'], subset['Predicted_Demand'], label=product, marker='o')

plt.title("📈 Forecasted Demand per Product (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Demand (Litres)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# !pip install matplotlib
# !pip install fpdf


# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
df = pd.read_csv("forecasted_demand_next_30_days.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Output path
pdf_path = "Forecast_Report_All_Products_With_Cover.pdf"

with PdfPages(pdf_path) as pdf:
    # --- Cover Page ---
    plt.figure(figsize=(11, 8))
    plt.axis('off')
    plt.title("📊 Demand Forecast Report\nFlavi Dairy Solutions", fontsize=20, weight='bold', pad=60)
    plt.text(0.5, 0.6, "Forecast Period: Next 30 Days\nProducts: Curd, Paneer, Cheese, Yogurt, etc.",
             ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.4, "Generated using AI-based ML Model (XGBoost)", ha='center', fontsize=12, style='italic')
    plt.text(0.5, 0.2, "Date: " + pd.Timestamp.now().strftime('%d %B %Y'), ha='center', fontsize=10)
    pdf.savefig()
    plt.close()

    # --- Product-wise Forecast Plots ---
    for product in df['Product'].unique():
        subset = df[df['Product'] == product]

        plt.figure(figsize=(10, 4))
        plt.plot(subset['Date'], subset['Predicted_Demand'], marker='o', color='teal')
        plt.title(f"Forecasted Demand for {product} - Next 30 Days")
        plt.xlabel("Date")
        plt.ylabel("Predicted Demand (Litres)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"✅ PDF report with cover page generated: {pdf_path}")


# %%
# Output path
excel_path = "Forecast_Report_By_Product.xlsx"

# Create Excel file with one sheet per product
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for product in df['Product'].unique():
        subset = df[df['Product'] == product]
        subset.to_excel(writer, sheet_name=product[:31], index=False)  # sheet name max 31 chars

print(f"✅ Excel report with one sheet per product saved as: {excel_path}")


# %% [markdown]
# @Dashboard

# %%
# !pip install ipywidgets openpyxl


# %%
# !pip install ipywidgets


# %%
# !pip install ipywidgets
# !jupyter labextension install @jupyter-widgets/jupyterlab-manager


# %%
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, FileLink
import ipywidgets as widgets

# Load data
df = pd.read_csv("forecasted_demand_next_30_days.csv")
df['Date'] = pd.to_datetime(df['Date'])
products = df['Product'].unique()

# Dropdown widget
product_dropdown = widgets.Dropdown(
    options=products,
    description='Select Product:',
    style={'description_width': 'initial'}
)

# Output widgets
output_plot = widgets.Output()
output_table = widgets.Output()
output_links = widgets.Output()

# Update function
def update_dashboard(change):
    product = product_dropdown.value
    filtered = df[df['Product'] == product]

    # Plot
    with output_plot:
        output_plot.clear_output()
        plt.figure(figsize=(10, 4))
        plt.plot(filtered['Date'], filtered['Predicted_Demand'], marker='o', color='teal')
        plt.title(f"Forecasted Demand - {product}")
        plt.xlabel("Date")
        plt.ylabel("Predicted Demand")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Table
    with output_table:
        output_table.clear_output()
        display(filtered[['Date', 'Predicted_Demand']])

    # Download links
    with output_links:
        output_links.clear_output()
        display(FileLink('Forecast_Report_All_Products_With_Cover.pdf', result_html_prefix="📄 PDF Report: "))
        display(FileLink('Forecast_Report_By_Product.xlsx', result_html_prefix="📊 Excel Report: "))

# Trigger update on change
product_dropdown.observe(update_dashboard, names='value')

# Initial run
update_dashboard(None)

# Display everything
display(product_dropdown, output_plot, output_table, output_links)


# %%
df.to_excel("Forecast_Report_All_Products.xlsx", index=False)


# %%
