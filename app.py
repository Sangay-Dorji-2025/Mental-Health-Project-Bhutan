import streamlit as st
import pandas as pd
import numpy as np

import os
st.title("Mental Health Stress Level Predictor with Visualizations (Bhutan)")
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

st.write("This section contains a minimal cleaning workflow. Modify it as needed.")

df_clean = df.copy()

# Placeholder: user will customize cleaning steps
# -------------------------------------------------
# Example steps (comment out or replace as needed)
df_clean.drop_duplicates(inplace=True)
df_clean.fillna(method="ffill", inplace=True)
df_clean.fillna(method="bfill", inplace=True)
# -------------------------------------------------

st.write("Cleaned dataset:")
st.dataframe(df_clean.head())

# ----------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDIT AS NEEDED)
# ----------------------------------------------------
st.header("3. Exploratory Data Analysis (EDA)")

st.write("Add your own analyses here. Below are optional placeholders.")

if st.checkbox("Show summary statistics"):
    st.write(df_clean.describe())

if st.checkbox("Show column info"):
    st.write(pd.DataFrame({
        "Column": df_clean.columns,
        "Dtype": df_clean.dtypes.astype(str)
    }))

# Optional chart
if st.checkbox("Show sample histogram"):
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col = st.selectbox("Choose a numeric column", numeric_cols)
        st.bar_chart(df_clean[col])
    else:
        st.warning("No numeric columns available.")
