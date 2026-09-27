# House Price Predictor

A regression model that predicts house prices from **location**, **number of rooms**, and **size (sqft)**, served through a Streamlit app.

## Project structure

```
house-price-predictor/
├── app.py                 # Streamlit app (the deployed entry point)
├── train_model.py         # Trains the regression model, saves model.pkl
├── generate_data.py       # Creates the sample dataset (data/house_prices.csv)
├── model.pkl              # Pre-trained model (already included, ready to deploy)
├── locations.pkl          # List of known locations for the dropdown
├── data/
│   └── house_prices.csv   # Sample training data
└── requirements.txt
```

The model is a `RandomForestRegressor` inside a scikit-learn `Pipeline` (one-hot encodes `location`, passes `rooms` and `size_sqft` through). On the included sample data it scores **R² ≈ 0.97** and **MAE ≈ $14k**.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use your own data

Replace `data/house_prices.csv` with your real listings (same three feature columns — `location`, `rooms`, `size_sqft` — plus a `price` column), then retrain:

```bash
python train_model.py
```

This overwrites `model.pkl` and `locations.pkl`. `app.py` doesn't need any changes.

## Deploy on Streamlit Community Cloud

1. **Push this folder to a public (or private) GitHub repo.** Include `model.pkl`, `locations.pkl`, and the `data/` folder — the app loads the pre-trained model directly, so no training happens on the server.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, and set the **main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and starts the app — no other configuration needed.
5. Any time you push new commits (e.g. a retrained `model.pkl`), the deployed app redeploys automatically.

That's it — the app is self-contained, so this is a "push and deploy" project with no extra setup steps or secrets required.
