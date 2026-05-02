import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal AI Factory", layout="wide")

st.title("🏭 The Universal AI Factory")
st.markdown("""
### Upload **ANY** dataset, and I will build a custom AI for it.
This tool automatically detects text vs. numbers, handles encoding, and trains a production-grade XGBoost model.
""")

# --- 1. DYNAMIC UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload Company Data (CSV)", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    st.write("### 1. Data Preview")
    st.dataframe(df.head())
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 2. SETTINGS (Sidebar) ---
    st.sidebar.header("Model Settings")
    
    # Select Target
    target_col = st.sidebar.selectbox("Select the Column to Predict (Target)", df.columns)
    
    # Select ID Columns to Drop
    drop_cols = st.sidebar.multiselect("Select ID columns to drop (e.g., CustomerID)", df.columns)

    # Button to Train
    if st.sidebar.button("🚀 Build Custom Model"):
        st.write("---")
        st.subheader("⚙️ Training in Progress...")
        
        # Progress Bar
        my_bar = st.progress(0)
        
        # --- A. CLEAN DATA ---
        df_clean = df.drop(columns=drop_cols)
        df_clean = df_clean.dropna()  # Simple dropna for demo
        
        # --- B. SEPARATE X and y ---
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # --- C. AUTO-DETECT & ENCODE ---
        # 1. Encode Features (X)
        X_encoded = X.copy()
        cat_cols = X.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            
        # 2. Encode Target (y) if it is text
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
        my_bar.progress(40)
        
        # --- D. TRAIN MODEL ---
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Calculate Scale Weight for Imbalance
        # (count of 0s / count of 1s)
        try:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        except:
            scale_pos_weight = 1 # Fallback if balanced or regression
            
        # Train XGBoost
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        my_bar.progress(80)
        
        # --- E. RESULTS ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        my_bar.progress(100)
        st.success(f"✅ Model Built Successfully! Accuracy: {acc:.2%}")
        
        # --- F. VISUALIZATIONS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            st.pyplot(fig)
            
        with col2:
            st.subheader("🏆 Feature Importance")
            importance = pd.Series(model.feature_importances_, index=X_encoded.columns)
            fig2, ax2 = plt.subplots()
            importance.nlargest(10).plot(kind='barh', ax=ax2, color='#4CAF50')
            ax2.set_title("Top 10 Drivers")
            st.pyplot(fig2)
            
        # --- G. DOWNLOAD ---
        st.subheader("📥 Download Predictions")
        results = X_test.copy()
        results['Actual'] = y_test
        results['Predicted'] = y_pred
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

else:
    st.info("Please upload a CSV file from the sidebar to begin.")