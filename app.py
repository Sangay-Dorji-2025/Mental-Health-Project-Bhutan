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
def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()

    # Handle >100 cases
    if x.startswith(">"):
        return float(x.replace(">", ""))

    # Remove bracket ranges (keep the first value only)
    if "[" in x:
        x = x.split("[")[0].strip()

    # Convert to float
    try:
        return float(x)
    except:
        return np.nan

df["Numeric"] = df["Numeric"].apply(clean_numeric)
df["Value"] = df["Value"].apply(clean_numeric)
df["Low"] = df["Low"].apply(clean_numeric)
df["High"] = df["High"].apply(clean_numeric)

# Clean year column
df["YEAR (DISPLAY)"] = pd.to_numeric(df["YEAR (DISPLAY)"], errors="coerce")

# Remove completely empty indicator rows
df_clean = df.dropna(subset=["Value", "YEAR (DISPLAY)"])


indicator = "Number of incident tuberculosis cases"  # choose your indicator

df_plot = df_clean[df_clean["GHO (DISPLAY)"] == indicator].copy()

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(df_plot["YEAR (DISPLAY)"], df_plot["Value"], marker="o", label="Value")
ax.fill_between(
    df_plot["YEAR (DISPLAY)"],
    df_plot["Low"],
    df_plot["High"],
    alpha=0.2,
    label="Confidence Interval"
)

ax.set_title(f"{indicator} Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()


indicator = st.sidebar.selectbox("Select Indicator", df["GHO (DISPLAY)"].unique())

df_filtered = df[df["GHO (DISPLAY)"] == indicator].sort_values("YEAR (DISPLAY)")

st.subheader(f"📌 {indicator}")
st.dataframe(df_filtered)

# Plot
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(df_filtered["YEAR (DISPLAY)"], df_filtered["Value"], marker="o", label="Value")

if df_filtered["Low"].notna().sum() > 0:
    ax.fill_between(
        df_filtered["YEAR (DISPLAY)"],
        df_filtered["Low"],
        df_filtered["High"],
        alpha=0.2,
        label="Confidence Interval"
    )

ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.set_title(indicator)
ax.legend()
st.pyplot(fig)

# ----------- Prediction Model ------------
if len(df_filtered) >= 3:
    st.subheader("📈 Prediction (Linear Regression)")
    
    X = df_filtered["YEAR (DISPLAY)"].values.reshape(-1,1)
    y = df_filtered["Value"].values

    model = LinearRegression()
    model.fit(X, y)

    next_year = df_filtered["YEAR (DISPLAY)"].max() + 1
    pred = model.predict([[next_year]])[0]

    st.write(f"Predicted value for **{next_year}**: **{pred:.2f}**")

else:
    st.warning("Not enough data points for prediction (need ≥3).")


df_pred = df_clean[df_clean["GHO (DISPLAY)"] == "Number of incident tuberculosis cases"].sort_values("YEAR (DISPLAY)")

X = df_pred["YEAR (DISPLAY)"].values.reshape(-1,1)
y = df_pred["Value"].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(2023, 2031).reshape(-1,1)
preds = model.predict(future_years)

pred_df = pd.DataFrame({"Year": future_years.flatten(), "Predicted TB Cases": preds})
print(pred_df)

# ----------------------------------------------------
# SECTION 7: EXPORT PROCESSED DATA (OPTIONAL)
# ----------------------------------------------------
st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

st.write("End of template. Modify each section to build your complete healthcare analytics workflow.")
