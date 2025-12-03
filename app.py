import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import joblib 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#from xgboost import XGBRegressor
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
    st.title("Linear Regression Dashboard — Bhutan Health Care")
    st.write("Sample Dataset")
    st.dataframe(df)

# ---------------------------------------------------------
# Build Linear Regression Model
# ---------------------------------------------------------
X = df_clean[["YEAR (DISPLAY)"]]
y = df_clean["Numeric"]

model = LinearRegression()
model.fit(X, y)

# Predictions
df_clean["Predicted"] = model.predict(X)

# ---------------------------------------------------------
# Calculate Metrics
# ---------------------------------------------------------
r2 = r2_score(y, df_clean["Predicted"])
mae = mean_absolute_error(y, df_clean["Predicted"])
rmse = np.sqrt(mean_squared_error(y, df_clean["Predicted"]))

# ---------------------------------------------------------
# Display Model Information
# ---------------------------------------------------------
st.write(" ## Model Summary")
# Coefficients explained
st.markdown(
    f"""
    <div style='text-align: left; font-size: 20px;'>
        <span>&#9679; Regression Equation: </span>   y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}<br>
        <span>&#9679; Slope (β₁): </span> {model.coef_[0]:.3f} → Increase per year<br>
        <span>&#9679; Intercept (β₀): </span> {model.intercept_:.3f} → Value when YEAR = 0
    </div>
    """,
    unsafe_allow_html=True
)
# Metrics
st.write(" ## Model Performance Metrics:")
st.markdown(
    f"""
    <div style='text-align: left; font-size: 20px;'>
        <p>&#9679; <strong>R Square (R²) Score:</strong> {r2:.3f}</p>
        <p>&#9679; <strong>Mean Absolute Error Score (MAE):</strong> {mae:.3f}</p>
        <p>&#9679; <strong>Root Mean Squared Error Score (RMSE):</strong> {rmse:.3f}</p>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------------------------------------------------
# Table of Actual vs Predicted
# ---------------------------------------------------------
st.write("##  Actual vs Predicted Table")
st.dataframe(df_clean)
#----------------------------------------------------------------
# SECTION 6: PREDICTION INTERFACE
# ----------------------------------------------------
st.header("6. Prediction Interface")

st.write("Build your prediction input widgets here.")
##################################################################
# --------------------------
# 5. PREDICT FUTURE YEAR
# --------------------------
st.write("### Predict for Future Year")
future_year = st.number_input("Enter future year:", min_value=2021, max_value=2050)

if st.button("Predict"):
    predicted_value = model.predict([[future_year]])
    st.success(f"Predicted Numeric value for {future_year}: **{predicted_value[0]:.2f}**") 
    

#################################################################################
# Placeholder: user defines input features
# Example: numeric inputs based on numeric columns
# prediction_inputs = {}
##############################################################################################
#if model is not None:
  #  st.subheader("Provide inputs for prediction")
#    
  #  numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
   # for col in numeric_cols:
    #    val = st.number_input(f"Input for {col}", float(df_features[col].min()), float(df_features[col].max()))
   #     prediction_inputs[col] = val
    
  #  if st.button("Predict"):
  #      try:
   #         X_input = pd.DataFrame([prediction_inputs])
   #         pred = model.predict(X_input)
    #        st.success(f"Model Prediction: {pred[0]}")
    #    except Exception as e:
    #        st.error(f"Prediction failed: {e}")
#else:
 #   st.warning("Model not available. Please train or upload a model.")

# ---------------------------------------------------------
# Plot Actual vs Predicted
# ---------------------------------------------------------
st.write(f"### 🔹Actual vs Predicted Plot")

fig, ax = plt.subplots(figsize=(9, 5))

# Scatter actual values
ax.scatter(df_clean["YEAR (DISPLAY)"], df_clean["Numeric"], s=80, label="Actual", alpha=0.8)

# Line predicted
ax.plot(df_clean["YEAR (DISPLAY)"], df_clean["Predicted"], linestyle='--', linewidth=2, label="Predicted")

# Make plot clean
ax.set_xlabel("YEAR (DISPLAY)", fontsize=12)
ax.set_ylabel("Numeric (Hospital Data)", fontsize=12)
ax.set_title("Linear Regression — Actual vs Predicted", fontsize=15, fontweight="bold")
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend()

st.pyplot(fig)
#####################################################################################################################
# ----------------------------------------------------
# SECTION 7: EXPORT PROCESSED DATA (OPTIONAL)
# ----------------------------------------------------
st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

st.write("End of template. Modify each section to build your complete healthcare analytics workflow.")
