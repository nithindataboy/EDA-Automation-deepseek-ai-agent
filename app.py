import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from io import BytesIO

# Suppress warnings for a clean experience
warnings.filterwarnings("ignore")

# Load DeepSeek API Key
DEEPSEEK_API_KEY = "ur deepseek api key"  # Replace with your valid API Key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/analyze"  # Ensure correct API endpoint

# Function to call DeepSeek AI for data insights
def get_deepseek_analysis(df):
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Convert dataframe to JSON for API
        data_json = df.to_json(orient="records")

        payload = {
            "dataset": data_json,
            "task": "exploratory_data_analysis"
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json().get("insights", "No insights found.")
        else:
            return f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}"
    except Exception as e:
        return f"API Request Failed: {str(e)}"

# Streamlit App Title
st.title("📊 AI-Powered EDA Agent (DeepSeek Enhanced)")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV only)", type=["csv"])

if uploaded_file:
    # Read Dataset
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Uploaded Data:")
    st.dataframe(df.head())

    # **Step 1: Data Cleaning**
    st.subheader("🛠️ Data Cleaning & Preprocessing")

    missing_before = df.isnull().sum().sum()  # Total missing values before

    # Handling Missing Values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical with mode
            else:
                df[col].fillna(df[col].mean(), inplace=True)  # Fill numeric with mean

    missing_after = df.isnull().sum().sum()  # Total missing values after

    # Display Data Cleaning Summary
    st.write(f"✅ **Missing Values Handled:** {missing_before} → {missing_after} (Now Clean!)")

    # **Step 2: Downloadable Cleaned CSV**
    st.subheader("📥 Download Cleaned Dataset")
    
    # Convert cleaned dataframe to CSV for download
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    cleaned_csv = convert_df_to_csv(df)

    st.download_button(
        label="📥 Download Cleaned CSV File",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    # **Step 3: Generate Automated EDA Report**
    st.subheader("📊 Automated EDA Report")

    eda_report = ProfileReport(df, explorative=True)
    eda_report_path = "eda_report.html"
    eda_report.to_file(eda_report_path)

    with open(eda_report_path, "rb") as f:
        st.download_button(label="📥 Download Full EDA Report", data=f, file_name="EDA_Report.html", mime="text/html")

    # **Step 4: Data Visualization**
    st.subheader("📈 Quick Data Visualizations")

    # Correlation Heatmap (Numeric Only)
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("⚠️ No numeric columns found for correlation heatmap.")

    # **Step 5: AI-Powered Insights from DeepSeek**
    st.subheader("🤖 AI-Based Insights from DeepSeek")
    deepseek_analysis = get_deepseek_analysis(df)
    st.write(deepseek_analysis)

    # **Step 6: AI-Based Model Recommendation**
    st.subheader("🧠 AI-Based Model Suggestion")

    # Check for target column
    target_column = st.selectbox("🎯 Select Target Column for Prediction (if any)", df.columns)

    if target_column:
        unique_values = df[target_column].nunique()
        
        if df[target_column].dtype == "object" or unique_values == 2:
            st.write("📌 Recommended Model: **Classification (e.g., RandomForest, XGBoost)**")
        else:
            st.write("📌 Recommended Model: **Regression (e.g., Linear Regression, XGBoost Regressor)**")

    st.success("🎉 EDA Analysis Completed Successfully!")
