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
# ----------------------------------------------------
# SECTION 7: EXPORT PROCESSED DATA (OPTIONAL)
# ----------------------------------------------------
st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

st.write("End of template. Modify each section to build your complete healthcare analytics workflow.")
