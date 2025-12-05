import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

st.set_page_config(page_title="Bhutan Healthcare Analytics", layout="wide")

st.title("Bhutan Healthcare Data Science Project")
st.write("Starter template for healthcare analytics and prediction using Streamlit.")

# -----------------------------
# SECTION 1: DATA LOADING
# -----------------------------
st.header("1. Load Healthcare Dataset")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is None:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("Preview of dataset:")
st.dataframe(df.head())

# -----------------------------
# SECTION 2: DATA CLEANING
# -----------------------------
st.header("2. Basic Data Cleaning")
df_clean = df.copy()

# Remove first row if unnecessary
df_clean = df_clean.drop(df_clean.index[0])
df_clean.drop_duplicates(inplace=True)
df_clean.fillna(method="ffill", inplace=True)
df_clean.fillna(method="bfill", inplace=True)

# Convert numeric-looking columns
cols_to_convert = {
    'float': ['Numeric', 'Low', 'High'],
    'int': ['YEAR (DISPLAY)', 'STARTYEAR', 'ENDYEAR']
}

for dtype, cols in cols_to_convert.items():
    for col in cols:
        if col in df_clean.columns:
            if dtype == 'float':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce', downcast='integer')

st.write("Cleaned dataset:")
st.dataframe(df_clean.head())

# -----------------------------
# SECTION 3: Exploratory Data Analysis
# -----------------------------
st.header("3. Exploratory Data Analysis (EDA)")
exclude_cols = ['YEAR (DISPLAY)', 'STARTYEAR', 'ENDYEAR']
numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

if st.checkbox("Show summary statistics"):
    st.write(df_clean[numeric_cols].describe())

if st.checkbox("Show column info"):
    st.write(pd.DataFrame({"Column": df_clean.columns, "Dtype": df_clean.dtypes.astype(str)}))

# -----------------------------
# SECTION 4: HISTOGRAMS
# -----------------------------
st.header("4. Histogram Analysis")

years = st.multiselect("Select Year(s)", sorted(df_clean["YEAR (DISPLAY)"].unique()))
df_filtered_hist = df_clean[df_clean["YEAR (DISPLAY)"].isin(years)] if years else df_clean.copy()
df_filtered_hist = df_filtered_hist.sort_values("YEAR (DISPLAY)")

col_selected = st.selectbox("Choose Column", ["Low", "Numeric", "High"])
bins = st.slider("Number of bins", 5, 100, 30)
log_scale = st.checkbox("Log Transformation")
use_kde = st.checkbox("Add KDE Curve")

data = df_filtered_hist[col_selected].dropna()
if log_scale:
    data = np.log1p(data)

fig, ax = plt.subplots(figsize=(8, 5))
if use_kde:
    sns.histplot(data, bins=bins, kde=True, ax=ax)
else:
    ax.hist(data, bins=bins, edgecolor="black")

mean_val, median_val, std_val = data.mean(), data.median(), data.std()
ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.2f}")
ax.axvline(median_val, color='blue', linestyle='-', label=f"Median: {median_val:.2f}")
ax.axvline(mean_val + std_val, color='green', linestyle=':', label=f"+1 Std: {mean_val + std_val:.2f}")
ax.set_title(f"Histogram of Bhutan Healthcare")
ax.set_xlabel(col_selected)
ax.set_ylabel("Frequency")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

st.subheader("Summary Statistics")
st.write(data.describe())

# -----------------------------
# SECTION 5: LINEAR REGRESSION MODEL
# -----------------------------
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
    st.title("Bhutan Health Care Data Science Project ")
    st.header("Model select is Supervised Learning Model (Linear Regression)")
    st.write(" ## Sample Dataset")
    st.dataframe(df_clean)

indicator_list = df_clean["GHO (DISPLAY)"].unique()
selected_indicator = st.selectbox("Choose Indicator", indicator_list)
df_indicator = df_clean[df_clean["GHO (DISPLAY)"] == selected_indicator].copy()

X = df_indicator[["YEAR (DISPLAY)"]]
y = df_indicator["Numeric"]

model = LinearRegression()
model.fit(X, y)
df_indicator["Predicted"] = model.predict(X)

# Metrics
r2 = r2_score(y, df_indicator["Predicted"])
mae = mean_absolute_error(y, df_indicator["Predicted"])
rmse = np.sqrt(mean_squared_error(y, df_indicator["Predicted"]))
#----------------------------------------------------
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
#------------------------------------------------------------------------------

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
# Plot Actual vs Predicted
years_plot = df_indicator["YEAR (DISPLAY)"].astype(int)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df_indicator["YEAR (DISPLAY)"], df_indicator["Numeric"], s=80, label="Actual")
ax.plot(df_indicator["YEAR (DISPLAY)"], df_indicator["Predicted"], color="red", linestyle='--', label="Predicted")
ax.set_xlabel("Year")
ax.set_ylabel("Numeric")
ax.set_title("Actual vs Predicted")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# -----------------------------
# SECTION 6: FUTURE PREDICTION
# -----------------------------
st.header("6. Predict Future Value")
future_year = st.number_input("Enter future year:", min_value=2030, max_value=2130)
if st.button("Predict"):
    pred_value = model.predict(np.array([[future_year]]))[0]
    st.success(f"Predicted Numeric value for {future_year}: {pred_value:.2f}")

# -----------------------------
# SECTION 7: EXPORT CLEANED DATA
# -----------------------------
st.header("7. Export Cleaned Data")
if st.button("Download Cleaned CSV"):
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")
