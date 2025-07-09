# %%
import pandas as pd
import random
from datetime import datetime, timedelta

# %%
# Parameters
start_date = datetime(2025, 5, 1)
days = 180
products = ['Curd', 'Yogurt premix', 'Fruit yogurt', 'paneer', 'mozzarella cheese', 'cheddar cheese','cottage cheese']

# %%
# Generate data
data = []
for i in range(days):
    date = start_date + timedelta(days=i)
    product = random.choice(products)
    temp = random.randint(28, 38)
    holiday = 1 if date.weekday() in [6] or random.random() < 0.05 else 0
    demand = random.randint(400, 1400)
    milk_supply = demand + random.randint(-100, 200)
    inventory = random.randint(100, 500)
    production = demand + random.randint(-50, 100)
    storage_util = random.randint(50, 90)
    downtime = round(random.uniform(0, 3), 1)

    data.append([date.strftime('%Y-%m-%d'), product, demand, milk_supply, inventory, production,
                 storage_util, downtime, temp, holiday])

# %%
# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Date", "Product", "Demand", "Milk_Supply", "Inventory", "Production",
    "Storage_Util (%)", "Downtime (hrs)", "Temperature (°C)", "Holiday"
])

# %%
# Save to CSV
df.to_csv("large_dairy_demand_forecast_sample.csv", index=False)
print("CSV file saved as 'large_dairy_demand_forecast_sample.csv'")

# %%
