import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

st.set_page_config(page_title="Bhutan Healthcare Analytics", layout="wide")

st.title("Bhutan Healthcare Data Science Project")
st.write("This is a starter template for a healthcare analytics and prediction application using Streamlit.")

# ----------------------------------------------------
# SECTION 1: DATA LOADING
# ----------------------------------------------------
st.header("1. Load Healthcare Dataset")

st.write("Upload your Bhutan healthcare dataset (CSV).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# ----------------------------------------------------
# SECTION 2: BASIC DATA CLEANING (EDIT AS NEEDED)
# ----------------------------------------------------
st.header("2. Basic Data Cleaning")
df_clean = df.copy()
indicator_list = sorted(df_clean["GHO (DISPLAY)"].unique())
selected_indicator = st.selectbox("Choose an Indicator:", indicator_list)

# Filter dataset
df_filtered = df_clean[df_clean["GHO (DISPLAY)"] == selected_indicator].copy()

st.write("### Filtered Data")
st.dataframe(df_filtered)

if df_filtered.shape[0] < 3:
    st.warning("⚠ Not enough data points for modeling. Need at least 3 years.")
    st.stop()

# ----------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDIT AS NEEDED)
# ----------------------------------------------------
st.header("3. Exploratory Data Analysis (EDA)")

st.write("Add your own analyses here. Below are optional placeholders.")


# ----------------------------------------------------
# SECTION 4: FEATURE ENGINEERING (USER FILLS IN)
# ----------------------------------------------------
###############################################################################
st.header("4. Feature Engineering -Histogram and advance features")

# ----------------------------------------------------
# SECTION 5: TRAIN OR LOAD MODEL (USER CHOOSES)
# ----------------------------------------------------
st.header("5. Machine Learning Model")

mode = st.radio("Choose model mode:", ["Load Existing Model", "Train New Model"])

if mode == "Load Existing Model":
    model_file = st.file_uploader("Upload trained .pkl model", type=["pkl"])
    if model_file is not None:
        model = joblib.load(model_file)
        st.success("Model loaded successfully.")
    else:
        st.warning("Upload a model to continue.")
        st.stop()

else:
    st.title("Bhutan Health Care Data Science(Supervised Learning Model:  Linear Regression)")
    st.write(" ## Sample Dataset")
    st.dataframe(df)

# ---------------------------------------------------------
# Build Linear Regression Model
# ---------------------------------------------------------

X = df_filtered[["YEAR (DISPLAY)"]]
y = df_filtered["Numeric"]

# Linear Regression
model = LinearRegression()
model.fit(X, y)

df_filtered["Predicted"] = model.predict(X)

# Metrics
r2 = r2_score(y, df_filtered["Predicted"])
mae = mean_absolute_error(y, df_filtered["Predicted"])
rmse = np.sqrt(mean_squared_error(y, df_filtered["Predicted"]))

# ---------------------------------------------------------
# DISPLAY MODEL SUMMARY
# ---------------------------------------------------------
st.write("### Model Summary")

st.markdown(
    f"""
    <div style='text-align:left; font-size:20px;'>
        <p>&#9679; <strong>Regression Equation:</strong> y = {model.coef_[0]:.3f} × Year + {model.intercept_:.3f}</p>
        <p>&#9679; <strong>Slope (β₁):</strong> {model.coef_[0]:.3f}</p>
        <p>&#9679; <strong>Intercept (β₀):</strong> {model.intercept_:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### Model Performance:")
st.markdown(
    f"""
    <div style='text-align:left; font-size:20px;'>
        <p>&#9679; <strong>R² Score:</strong> {r2:.3f}</p>
        <p>&#9679; <strong>MAE:</strong> {mae:.3f}</p>
        <p>&#9679; <strong>RMSE:</strong> {rmse:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SHOW TABLE
# ---------------------------------------------------------
st.write("### Actual vs Predicted Table")
st.dataframe(df_filtered[["YEAR (DISPLAY)", "Numeric", "Predicted"]])

future_year = st.number_input(
    "Enter future year to predict:",
    min_value=int(df_filtered["YEAR (DISPLAY)"].max()),
    max_value=2200
)

if st.button("Predict Future Value"):
    future_value = model.predict([[future_year]])[0]
    st.success(f"Predicted value for {future_year}: **{future_value:.2f}**")

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------
st.header("4. Actual vs Predicted Plot")

fig, ax = plt.subplots(figsize=(9, 5))

# Scatter actual
ax.scatter(df_filtered["YEAR (DISPLAY)"], df_filtered["Numeric"], 
           label="Actual", s=80, alpha=0.8)

# Line predicted
ax.plot(df_filtered["YEAR (DISPLAY)"], df_filtered["Predicted"], 
        linestyle='--', linewidth=2, color='red', label="Predicted")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel(selected_indicator, fontsize=12)
ax.set_title(f"Trend for: {selected_indicator}", fontsize=15, fontweight="bold")
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

st.pyplot(fig)
# ----------------------------------------------------
# SECTION 7: EXPORT PROCESSED DATA (OPTIONAL)
# ----------------------------------------------------
st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

st.write("End of template. Modify each section to build your complete healthcare analytics workflow.")
