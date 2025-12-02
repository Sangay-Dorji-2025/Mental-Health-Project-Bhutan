import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import joblib 
import seaborn as sns
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
#st.header("4. Feature Engineering")

#st.write("Create engineered features here. Add your own logic below.")

# Placeholder
#df_features = df_clean.copy()

#st.write("Feature-engineered data preview:")
#st.dataframe(df_features.head())
#############################################################################
st.header("4. Feature Engineering")
st.header("📊 Histogram Viewer")

# -------------------------------
# 1️⃣ STEP 1 — SIMPLE HISTOGRAM
# -------------------------------

if st.checkbox("Show Simple Histogram"):

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        col = st.selectbox("Choose a numeric column", numeric_cols, key="simple_hist")

        # Simple histogram (no settings)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df_clean[col].dropna(), bins=30, edgecolor="black")

        ax.set_title(f"Simple Histogram of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        st.pyplot(fig)

    else:
        st.warning("No numeric columns available.")

# -----------------------------------
# 2️⃣ STEP 2 — ADVANCED OPTIONS
# -----------------------------------

st.subheader("⚙️ Feature Engineering Options")

if st.checkbox("Enable Advanced Histogram Options"):

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:

        col = st.selectbox("Choose a numeric column", numeric_cols, key="advanced_hist")

        # Choose number of bins
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        # KDE option
        use_kde = st.checkbox("Add KDE Curve (Smooth Density)")

        # Log transform
        log_scale = st.checkbox("Apply Log Transformation")

        # Apply log transform if chosen
        data = df_clean[col].dropna()
        if log_scale:
            data = np.log1p(data)

        # ADVANCED HISTOGRAM
        fig, ax = plt.subplots(figsize=(8,5))

        if use_kde:
            sns.histplot(data, bins=bins, kde=True, edgecolor="black", ax=ax)
        else:
            ax.hist(data, bins=bins, edgecolor="black")

        # Add statistic lines
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.3, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color='blue', linestyle='-', linewidth=1.3, label=f"Median: {median_val:.2f}")
        ax.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=1.3, label=f"+1 Std: {mean_val+std_val:.2f}")
        ax.axvline(mean_val - std_val, color='green', linestyle=':', linewidth=1.3, label=f"-1 Std: {mean_val-std_val:.2f}")

        ax.set_title(f"Advanced Histogram of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend()

        st.pyplot(fig)

        # Summary statistics
        st.write("### 📘 Summary Statistics")
        st.write(data.describe())

        # Optional boxplot
        if st.checkbox("Show Boxplot"):
            fig2, ax2 = plt.subplots()
            ax2.boxplot(data)
            ax2.set_title(f"Boxplot of {col}")
            ax2.set_ylabel(col)
            st.pyplot(fig2)

    else:
        st.warning("No numeric columns available.")

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
    st.write("Add your model training code below.")
    st.info("This template does not include a training implementation. Write your own training logic here.")

    # Placeholder to prevent execution errors
    model = None

# ----------------------------------------------------
# SECTION 6: PREDICTION INTERFACE
# ----------------------------------------------------
st.header("6. Prediction Interface")

st.write("Build your prediction input widgets here.")

# Placeholder: user defines input features
# Example: numeric inputs based on numeric columns
prediction_inputs = {}

if model is not None:
    st.subheader("Provide inputs for prediction")
    
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        val = st.number_input(f"Input for {col}", float(df_features[col].min()), float(df_features[col].max()))
        prediction_inputs[col] = val
    
    if st.button("Predict"):
        try:
            X_input = pd.DataFrame([prediction_inputs])
            pred = model.predict(X_input)
            st.success(f"Model Prediction: {pred[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("Model not available. Please train or upload a model.")

# ----------------------------------------------------
# SECTION 7: EXPORT PROCESSED DATA (OPTIONAL)
# ----------------------------------------------------
st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

st.write("End of template. Modify each section to build your complete healthcare analytics workflow.")
