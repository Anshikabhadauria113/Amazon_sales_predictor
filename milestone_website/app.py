import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression

# Page Setup
st.set_page_config(page_title="Sales Predictor")
st.title("📊 Amazon Sales Prediction Model")

# 1. File Upload Section
st.write("### Step 1: Upload your CSV file")
uploaded_file = st.file_uploader("Apni CSV file yahan dalein", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File Successfully Uploaded!")

    # Data Filtering
    try:
        # Sirf Monthly Trends wala data lenge
        # Note: Ensure your CSV actually has 'Report_Section' column
        if 'Report_Section' in df.columns:
            data = df[df['Report_Section'] == 'MONTHLY_SALES_TRENDS'].copy()
        else:
            # Fallback if column missing, assumes entire sheet is relevant data
            data = df.copy() 
        
        # Rename Columns (Metrics set kar rahe hain)
        # Check if columns exist before renaming to avoid errors
        data = data.rename(columns={
            'Dimension': 'Month',
            'Metric2': 'Units_Sold',
            'Metric3': 'Avg_Price'
        })
        
        # Missing values hatana
        data = data.dropna(subset=['Units_Sold', 'Avg_Price'])

        # Classification Logic (High vs Low)
        median_sales = data['Units_Sold'].median()
        data['Status'] = np.where(data['Units_Sold'] > median_sales, 'High Sales 🟢', 'Low Sales 🔴')
        data['Target'] = np.where(data['Units_Sold'] > median_sales, 1, 0)

        # 2. Models Banana (Regression & Classification)
        X = data[['Avg_Price']]
        y_reg = data['Units_Sold']
        y_clf = data['Target']

        reg = LinearRegression().fit(X, y_reg)
        clf = LogisticRegression().fit(X, y_clf)
        
        influence = reg.coef_[0]

        # 3. Results Dikhana
        st.divider()
        st.write("### 📉 Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Price Influence: Har ₹1000 badhane par sales **{abs(influence*1000):.0f} units** {'kam' if influence < 0 else 'zyada'} hoti hai.")
        with col2:
            st.metric("Average Price", f"₹{data['Avg_Price'].mean():,.0f}")

        # 4. Graphs
        st.write("### 📊 Visuals")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sns.regplot(x='Avg_Price', y='Units_Sold', data=data, ax=ax[0], color='red')
        ax[0].set_title("Price vs Sales Impact")
        
        sns.boxplot(x='Status', y='Avg_Price', data=data, ax=ax[1])
        ax[1].set_title("High vs Low Sales Price Range")
        st.pyplot(fig)

        # 5. Prediction Tool
        st.divider()
        st.write("### 🔮 Future Predictor")
        price_input = st.number_input("Naya Price Daalo (₹)", value=75000, step=1000)
        
        if st.button("Predict Karo"):
            # FIXED: Removed the extra '+' inside the predict function
            pred_sales = reg.predict([[price_input]])[0]
            pred_class = clf.predict([[price_input]])[0]
            status = "High Sales 🟢" if pred_class == 1 else "Low Sales 🔴"
            
            st.success(f"Agar price ₹{price_input} rakha, toh approx **{int(pred_sales)} units** bikenge.")
            st.info(f"Category: **{status}**")

    except Exception as e:
        st.error(f"Error: {e}. Please upload the correct Amazon CSV file.")