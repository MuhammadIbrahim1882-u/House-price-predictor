"""
Trains a regression model to predict house prices from
location, number of rooms, and size (sqft), then saves the
fitted pipeline to model.pkl.

Run: python train_model.py
"""
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/house_prices.csv"
MODEL_PATH = "model.pkl"

df = pd.read_csv(DATA_PATH)

X = df[["location", "rooms", "size_sqft"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ],
    remainder="passthrough",  # keeps rooms, size_sqft as-is
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42
    )),
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE:  ${mae:,.0f}")
print(f"R^2:  {r2:.3f}")

joblib.dump(model, MODEL_PATH)
print(f"Saved trained model to {MODEL_PATH}")

# Also save the list of known locations so the app can build
# its dropdown without needing to read the whole CSV.
locations = sorted(df["location"].unique().tolist())
joblib.dump(locations, "locations.pkl")
