import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    locations = joblib.load("locations.pkl")
    return model, locations


model, locations = load_model()

st.title("🏠 House Price Predictor")
st.write(
    "Predict an estimated house price from **location**, **number of rooms**, "
    "and **size**, using a Random Forest regression model."
)

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Location", locations)
        rooms = st.number_input("Number of rooms (bedrooms)", min_value=1, max_value=15, value=3, step=1)
    with col2:
        size_sqft = st.number_input("Size (sqft)", min_value=200, max_value=10000, value=1200, step=50)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_df = pd.DataFrame(
        [[location, rooms, size_sqft]],
        columns=["location", "rooms", "size_sqft"],
    )
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Price: ${prediction:,.0f}")

    st.caption(
        "This is trained on a synthetic sample dataset (data/house_prices.csv). "
        "Swap that file for real listings and re-run train_model.py to make "
        "predictions reflect a real market."
    )

st.divider()
st.subheader("Explore the training data")
df = pd.read_csv("data/house_prices.csv")
show_location = st.selectbox("Filter by location (optional)", ["All"] + locations, key="filter")
if show_location != "All":
    df = df[df["location"] == show_location]
st.dataframe(df.head(50), use_container_width=True)
st.scatter_chart(df, x="size_sqft", y="price", color="location")
