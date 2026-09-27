"""
Generates a synthetic-but-realistic house price dataset.
Features: location, rooms (bedrooms), size_sqft
Target: price

Run this once to (re)create data/house_prices.csv.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Locations with different base price levels (per sqft) — swap these
# for real city/neighborhood names for your own data.
LOCATIONS = {
    "Downtown":     220,
    "Suburb North": 140,
    "Suburb South": 130,
    "Uptown":       190,
    "Rural":         85,
}

N = 1200
rows = []
for _ in range(N):
    location = np.random.choice(list(LOCATIONS.keys()))
    rooms = np.random.randint(1, 7)  # 1 to 6 bedrooms
    size_sqft = int(np.random.normal(loc=400 + rooms * 250, scale=150))
    size_sqft = max(300, size_sqft)

    base_rate = LOCATIONS[location]
    price = (
        size_sqft * base_rate
        + rooms * 8000
        + np.random.normal(0, 15000)  # noise
    )
    price = max(20000, round(price, -2))  # floor + round to nearest 100

    rows.append([location, rooms, size_sqft, price])

df = pd.DataFrame(rows, columns=["location", "rooms", "size_sqft", "price"])
df.to_csv("data/house_prices.csv", index=False)
print(f"Wrote {len(df)} rows to data/house_prices.csv")
print(df.head())
