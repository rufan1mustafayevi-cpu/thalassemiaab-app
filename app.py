import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thal-AI Classifier | Clinical Diagnostic Tool",
    page_icon="🩸",
    layout="wide"
)

# ---------------------------------------------------------
# HELPER FUNCTIONS & MODEL TRAINING
# ---------------------------------------------------------
def engineer_features(df):
    X = df.copy()
    X['mentzer_index'] = X['mcv'] / X['rbc']
    X['shine_lal_index'] = (X['mcv']**2 * X['mch']) / 100.0
    X['srivastava_index'] = X['mch'] / X['rbc']
    X['green_king_index'] = (X['mcv']**2 * X['rdw']) / (100.0 * X['hb'])
    X['ehsani_index'] = X['mcv'] - (10.0 * X['rbc'])
    X['england_fraser_index'] = X['mcv'] - X['rbc'] - (5.0 * X['hb']) - 3.4
    X['mch_mcv_ratio'] = X['mch'] / X['mcv']
    X['mcv_rbc_product'] = np.log1p(X['mcv'] * X['rbc'])
    return X

@st.cache_resource
def load_and_train_model():
    alpha_file = 'alphanorm (2).csv'
    beta_file = 'thalassimia.xlsx - Sheet1.csv'
    
    if not os.path.exists(alpha_file) or not os.path.exists(beta_file):
        st.error("⚠️ Dataset files missing! Ensure 'alphanorm (2).csv' and 'thalassimia.xlsx - Sheet1.csv' are in the directory.")
        st.stop()
        
    # Load Alpha dataset
    df_alpha = pd.read_csv(alpha_file)
    df_alpha.columns = df_alpha.columns.str.strip().str.lower()
    df_alpha_clean = df_alpha[['mcv', 'mch', 'rbc', 'hb', 'rdw', 'phenotype']].copy()
    df_alpha_clean.dropna(inplace=True)

    # Load Beta dataset
    df_beta = pd.read_csv(beta_file)
    df_beta.columns = df_beta.columns.str.strip().str.lower()
    df_beta.rename(columns={'hbg': 'hb'}, inplace=True)
    df_beta['hba2'] = pd.to_numeric(df_beta['hba2'], errors='coerce')
    df_beta_clean = df_beta[['mcv', 'mch', 'rbc', 'hb', 'hba2']].copy()
    df_beta_clean.dropna(subset=['mcv', 'mch', 'rbc', 'hb', 'hba2'], inplace=True)

    df_beta_clean['phenotype'] = df_beta_clean.apply(
        lambda r: 'beta carrier' if r['hba2'] >= 3.5 else 'normal', axis=1
    )
    df_beta_clean.drop(columns=['hba2'], inplace=True)

    if 'rdw' not in df_beta_clean.columns:
        df_beta_clean['rdw'] = 14.5

    # Combine
    dataset = pd.concat([df_alpha_clean, df_beta_clean], ignore_index=True)
    num_cols = ['mcv', 'mch', 'rbc', 'hb', 'rdw']
    for col in num_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    dataset.dropna(inplace=True)

    # Feature engineering & training
    X_raw = dataset[['mcv', 'mch', 'rbc', 'hb', 'rdw']]
    X_eng = engineer_features(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(dataset['phenotype'])

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_eng, y)

    return model, le, X_eng.columns

# Load model and encoders
model, le, feature_columns = load_and_train_model()

# ---------------------------------------------------------
# NAVIGATION & HEADER
# ---------------------------------------------------------
st.title("🩸 Thal-AI: Clinical Diagnostic Classifier")
st.markdown("Automated Differentiation of **α-Thalassemia Trait**, **β-Thalassemia Trait**, and **Normal Phenotypes** using Machine Learning & CBC Indices.")

tab1, tab2, tab3 = st.tabs(["🔬 Single Patient Predictor", "📊 Batch Processing (CSV)", "📈 Model Interpretability & XAI"])

# ---------------------------------------------------------
# TAB 1: SINGLE PATIENT DIAGNOSIS
# ---------------------------------------------------------
with tab1:
    st.subheader("Patient Complete Blood Count (CBC) Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mcv = st.number_input("MCV (fL)", min_value=40.0, max_value=120.0, value=68.7, step=0.1)
        hb = st.number_input("Hemoglobin - Hb (g/dL)", min_value=3.0, max_value=20.0, value=10.8, step=0.1)
        
    with col2:
        mch = st.number_input("MCH (pg)", min_value=10.0, max_value=40.0, value=21.2, step=0.1)
        rdw = st.number_input("RDW (%)", min_value=8.0, max_value=30.0, value=13.4, step=0.1)
        
    with col3:
        rbc = st.number_input("RBC Count (10^6/µL)", min_value=1.0, max_value=10.0, value=5.12, step=0.01)

    # Predict Button
    if st.button("Run Diagnostic Inference", type="primary", use_container_width=True):
        # Construct single row dataframe
        patient_raw = pd.DataFrame([{
            'mcv': mcv, 'mch': mch, 'rbc': rbc, 'hb': hb, 'rdw': rdw
        }])
        
        # Calculate engineered features
        patient_eng = engineer_features(patient_raw)
        
        # Inference
        pred_idx = model.predict(patient_eng)[0]
        probs = model.predict_proba(patient_eng)[0]
        pred_label = le.inverse_transform([pred_idx])[0].upper()

        st.markdown("---")
        st.subheader("Diagnostic Results")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if "ALPHA" in pred_label:
                st.error(f"### Predicted Diagnosis:\n**{pred_label}**")
            elif "BETA" in pred_label:
                st.warning(f"### Predicted Diagnosis:\n**{pred_label}**")
            else:
                st.success(f"### Predicted Diagnosis:\n**{pred_label}**")
                
            st.caption("⚡ Model decision powered by Random Forest + Derived Hematology Indices.")

        with res_col2:
            # Probability Bar Chart
            prob_df = pd.DataFrame({
                'Class': [c.upper() for c in le.classes_],
                'Probability': probs * 100
            })
            fig_prob = px.bar(
                prob_df, x='Probability', y='Class', orientation='h',
                color='Class', text_auto='.1f',
                title="Model Classification Confidence (%)",
                color_discrete_map={'ALPHA CARRIER': '#EF553B', 'BETA CARRIER': '#FECB52', 'NORMAL': '#00CC96'}
            )
            fig_prob.update_layout(showlegend=False, xaxis_range=[0, 100], height=250)
            st.plotly_chart(fig_prob, use_container_width=True)

        # Derived Indices Table
        st.subheader("Calculated Clinical Discrimination Indices")
        idx_col1, idx_col2, idx_col3, idx_col4, idx_col5, idx_col6 = st.columns(6)
        
        idx_col1.metric("Mentzer Index", f"{patient_eng['mentzer_index'][0]:.2f}", delta="< 13 suggests Trait")
        idx_col2.metric("Green & King", f"{patient_eng['green_king_index'][0]:.2f}", delta="< 65 suggests Trait")
        idx_col3.metric("Shine & Lal", f"{patient_eng['shine_lal_index'][0]:.2f}", delta="< 1530 suggests Trait")
        idx_col4.metric("Srivastava", f"{patient_eng['srivastava_index'][0]:.2f}", delta="< 3.8 suggests Trait")
        idx_col5.metric("Ehsani", f"{patient_eng['ehsani_index'][0]:.2f}", delta="< 15 suggests Trait")
        idx_col6.metric("England & Fraser", f"{patient_eng['england_fraser_index'][0]:.2f}", delta="< 0 suggests Trait")

# ---------------------------------------------------------
# TAB 2: BATCH CSV PROCESSING
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload Patient Batch Dataset")
    st.write("Upload a CSV file containing columns: `mcv`, `mch`, `rbc`, `hb`, `rdw`")
    
    uploaded_batch = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_batch is not None:
        batch_df = pd.read_csv(uploaded_batch)
        batch_df.columns = batch_df.columns.str.strip().str.lower()
        
        req_cols = ['mcv', 'mch', 'rbc', 'hb', 'rdw']
        if all(col in batch_df.columns for col in req_cols):
            batch_eng = engineer_features(batch_df[req_cols])
            preds = model.predict(batch_eng)
            batch_df['predicted_phenotype'] = le.inverse_transform(preds)
            
            st.success(f"Successfully processed {len(batch_df)} patient records!")
            st.dataframe(batch_df, use_container_width=True)
            
            # Download Results
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv_data,
                file_name="thalassemia_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Missing required columns. CSV must contain: {req_cols}")

# ---------------------------------------------------------
# TAB 3: EXPLAINABLE AI & FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab3:
    st.subheader("Gini Feature Importance Analysis")
    st.write("Explains which engineered indices contribute most to differentiating $\\alpha$-thalassemia from $\\beta$-thalassemia.")
    
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
    
    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        labels={'x': 'Relative Importance Score', 'y': 'Feature / Calculated Index'},
        title="Random Forest Feature Contributions",
        color=importances.values,
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(height=500)
    st.plotly_chart(fig_imp, use_container_width=True)