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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
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

st.write("This section mainly perform folloiwng task: a.Make a copy of original Dataframe. b. Remove duplicates.c. Handle NaN values. d. Converts numeric-looking object columns to float")
 # Make a copy of the original DataFrame
df_clean = df.copy()

# Placeholder: user will customize cleaning steps
# -------------------------------------------------
# Example steps (comment out or replace as needed)
# Remove the first row of DataFrame seems like its not necessary
df_clean = df_clean.drop(df_clean.index[0])
# Remove duplicates
df_clean.drop_duplicates(inplace=True)
# Handle NaN: forward fill then backward fill
df_clean.fillna(method="ffill", inplace=True)
df_clean.fillna(method="bfill", inplace=True)
# Convert numeric-looking object columns to float and integer coercing errors to NaN
cols_to_convert = {
    'float': ['Numeric', 'Low', 'High'],
    'int': ['YEAR (DISPLAY)', 'STARTYEAR', 'ENDYEAR']
}
for dtype, cols in cols_to_convert.items():
    for col in cols:
        if dtype == 'float':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        elif dtype == 'int':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce', downcast='integer')
            
# -------------------------------------------------
st.write("Cleaned dataset:")
st.dataframe(df_clean.head())
# Already displayed in section 3
#st.dataframe(df_clean.dtypes)

# ----------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDIT AS NEEDED)
# ----------------------------------------------------
st.header("3. Exploratory Data Analysis (EDA)")

st.write("Add your own analyses here. Below are optional placeholders.")

exclude_cols = ['YEAR (DISPLAY)', 'STARTYEAR', 'ENDYEAR']
#numeric_cols = df_clean.select_dtypes(include='number').columns
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
if st.checkbox("Show summary statistics"):
    st.write(df_clean[numeric_cols].describe())

if st.checkbox("Show column info"):
    st.write(pd.DataFrame({
        "Column": df_clean.columns,
        "Dtype": df_clean.dtypes.astype(str)
    }))
# ----------------------------------------------------
# SECTION 4: FEATURE ENGINEERING (USER FILLS IN)
# ----------------------------------------------------
###############################################################################
st.header("4. Feature Engineering -Histogram and advance features")

# =====================================================
# SECTION 1 — SIMPLE HISTOGRAM (YEAR sorted when chosen)
# =====================================================
st.header("4.1 Simple Histogram")
#numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()

if st.checkbox("Show Simple Histogram"):
    if numeric_cols:
        col = st.selectbox("Choose a numeric column", numeric_cols)

        # If YEAR selected → sort ascending before plotting
        if col == "YEAR (DISPLAY)":
            df_sorted = df_clean.sort_values(by="YEAR (DISPLAY)")
            data_simple = df_sorted[col]
            xlabel = "YEAR (DISPLAY) – sorted ascending"
        else:
            data_simple = df_clean[col]
            xlabel = col

        fig_simple, ax_simple = plt.subplots()
        ax_simple.hist(data_simple, bins=20)
        ax_simple.set_xlabel(xlabel)
        ax_simple.set_ylabel("Frequency")
        ax_simple.set_title(f"Histogram of Bhutan Health Care")

        st.pyplot(fig_simple)

    else:
        st.warning("No numeric columns available.")


# =====================================================
# SECTION 2 — ADVANCED FEATURE ENGINEERING HISTOGRAM
# =====================================================

st.header("4.2 Histogram (Advanced Feature Engineering) ")

# --- Year selection ---
years = st.multiselect(
    "Select Year(s) to display",
    sorted(df_clean["YEAR (DISPLAY)"].unique())   # sorted ascending
)

# Filter by year
if years:
    df_filtered = df_clean[df_clean["YEAR (DISPLAY)"].isin(years)]
else:
    df_filtered = df_clean.copy()

# Always sort by year
df_filtered = df_filtered.sort_values(by="YEAR (DISPLAY)", ascending=True)

# --- Column selection ---
col_options = ["Low", "Numeric", "High"]
col_selected = st.selectbox("Choose a column", col_options)

# --- Advanced options ---
bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
log_scale = st.checkbox("Apply Log Transformation")
use_kde = st.checkbox("Add KDE Curve (Smooth Density)")

# --- Prepare data ---
data = df_filtered[col_selected].dropna()

if log_scale:
    data = np.log1p(data)

# --- Plot histogram ---
fig, ax = plt.subplots(figsize=(8,5))

if use_kde:
    sns.histplot(data, bins=bins, kde=True, edgecolor="black", ax=ax)
else:
    ax.hist(data, bins=bins, edgecolor="black")

# --- Mean / Median / Std lines ---
mean_val = data.mean()
median_val = data.median()
std_val = data.std()

ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.3, label=f"Mean: {mean_val:.2f}")
ax.axvline(median_val, color='blue', linestyle='-', linewidth=1.3, label=f"Median: {median_val:.2f}")
ax.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=1.3, label=f"+1 Std: {mean_val+std_val:.2f}")
# ax.axvline(mean_val - std_val, color='yellow', linestyle=':', linewidth=1.3, label=f"-1 Std: {mean_val-std_val:.2f}")

# --- Labels ---
if years:
    year_text = ", ".join(map(str, years))   # selected years
else:
    year_text = "All Years"

ax.set_title(
    f"Histogram of Bhutan Health Care ({year_text})",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel(col_selected)
ax.set_ylabel("Frequency")
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend()

st.pyplot(fig)

# --- Summary statistics ---
st.subheader("Summary Statistics")
st.write(data.describe())

# --- Optional boxplot ---
if st.checkbox("Show Boxplot"):
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.boxplot(data)
    ax2.set_title("Boxplot of Bhutan Health Care")
    ax2.set_ylabel(col_selected)
    st.pyplot(fig2)

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
#numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
#target_col = st.selectbox("Select Target Column", numeric_cols)
#feature_cols = st.multiselect("Select Feature Columns", [c for c in numeric_cols if c != target_col], default=[c for c in numeric_cols if c != target_col])
X = df_clean[["YEAR (DISPLAY)"]]
y = df_clean["Numeric"]
#if feature_cols:
   # X = df_clean[feature_cols]
   # y = df_clean[target_col]

 # --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # --- Model Selection ---
 model_option = st.selectbox("Select Regression Model", ["Linear Regression", "Polynomial Regression (deg 2)", "Random Forest"])
    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "Polynomial Regression (deg 2)":
         model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

# --- Fit Model ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 # --- Flatten predictions to 1D array to avoid TypeError ---
        y_pred = np.ravel(y_pred)
        y_test = np.ravel(y_test)

 # --- Metrics ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MAE:** {mae:.3f}, **RMSE:** {rmse:.3f}, **R² Score:** {r2:.3f}")
        

        # --- Add predictions to full df for plotting ---
        df_clean["Predicted"] = model.predict(df_clean[feature_cols])

        # --- Plot Actual vs Predicted ---
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df.index, df_clean[target_col], s=100, alpha=0.8, edgecolor="black", label="Actual", color="blue")
        ax.plot(df.index, df_clean["Predicted"], color="red", marker='o', linewidth=2, markersize=6, label="Predicted")
        ax.set_title(f"Actual vs Predicted - {model_option}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel(target_col, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.35)
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
