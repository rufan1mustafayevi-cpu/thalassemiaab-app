import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Thal-AI | Clinical Diagnostic Tool",
    page_icon="🩸",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING ENGINE
# ---------------------------------------------------------
def engineer_features(df):
    X = df.copy()
    # Established Clinical Discrimination Indices
    X['mentzer_index'] = X['mcv'] / X['rbc']
    X['shine_lal_index'] = (X['mcv']**2 * X['mch']) / 100.0
    X['srivastava_index'] = X['mch'] / X['rbc']
    X['green_king_index'] = (X['mcv']**2 * X['rdw']) / (100.0 * X['hb'])
    X['ehsani_index'] = X['mcv'] - (10.0 * X['rbc'])
    X['england_fraser_index'] = X['mcv'] - X['rbc'] - (5.0 * X['hb']) - 3.4
    
    # Ratios & Interactions
    X['mch_mcv_ratio'] = X['mch'] / X['mcv']
    X['mcv_rbc_product'] = np.log1p(X['mcv'] * X['rbc'])
    return X

# ---------------------------------------------------------
# 3. DYNAMIC DATA LOADING & MODEL TRAINING
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    all_files = os.listdir('.')
    
    # Locate files dynamically to prevent filename casing/spacing errors
    alpha_file = next((f for f in all_files if 'alpha' in f.lower() and f.endswith('.csv')), None)
    beta_file = next((f for f in all_files if 'thal' in f.lower() and (f.endswith('.csv') or f.endswith('.xlsx'))), None)
    
    if not alpha_file or not beta_file:
        return None, None, None, f"Dataset files missing in directory. Found: {all_files}"
        
    try:
        # Load Alpha dataset
        df_alpha = pd.read_csv(alpha_file)
        df_alpha.columns = df_alpha.columns.str.strip().str.lower()
        df_alpha_clean = df_alpha[['mcv', 'mch', 'rbc', 'hb', 'rdw', 'phenotype']].copy().dropna()

        # Load Beta dataset (handles both CSV and Excel)
        if beta_file.endswith('.xlsx'):
            df_beta = pd.read_excel(beta_file)
        else:
            df_beta = pd.read_csv(beta_file)
            
        df_beta.columns = df_beta.columns.str.strip().str.lower()
        df_beta.rename(columns={'hbg': 'hb'}, inplace=True)
        df_beta['hba2'] = pd.to_numeric(df_beta['hba2'], errors='coerce')
        
        # Filter rows with complete core values
        df_beta_clean = df_beta[['mcv', 'mch', 'rbc', 'hb', 'hba2']].copy().dropna()

        # Clinical labeling rule: HbA2 >= 3.5% = Beta Carrier
        df_beta_clean['phenotype'] = df_beta_clean.apply(
            lambda r: 'beta carrier' if r['hba2'] >= 3.5 else 'normal', axis=1
        )
        df_beta_clean.drop(columns=['hba2'], inplace=True)

        if 'rdw' not in df_beta_clean.columns:
            df_beta_clean['rdw'] = 14.5  # Baseline median population assumption

        # Merge datasets
        dataset = pd.concat([df_alpha_clean, df_beta_clean], ignore_index=True)
        for col in ['mcv', 'mch', 'rbc', 'hb', 'rdw']:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        dataset.dropna(inplace=True)

        # Train model
        X_raw = dataset[['mcv', 'mch', 'rbc', 'hb', 'rdw']]
        X_eng = engineer_features(X_raw)

        le = LabelEncoder()
        y = le.fit_transform(dataset['phenotype'])

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_eng, y)

        return model, le, X_eng.columns, None

    except Exception as e:
        return None, None, None, str(e)

# Initialize model
model, le, feature_columns, err = load_and_train_model()

# ---------------------------------------------------------
# 4. USER INTERFACE
# ---------------------------------------------------------
st.markdown('<div class="main-title">🩸 Thal-AI: Clinical Diagnostic Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Classification for α-Thalassemia Trait, β-Thalassemia Trait, and Normal Controls</div>', unsafe_allow_html=True)

if err:
    st.error(f"⚠️ Error loading model: {err}")
    st.info("Make sure `alphanorm (2).csv` and `thalassimia.xlsx - Sheet1.csv` are committed to your GitHub repository root folder.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔬 Single Patient Predictor", "📊 Batch Processing (CSV)", "📈 Model Interpretability (XAI)"])

# ---------------------------------------------------------
# TAB 1: SINGLE PATIENT PREDICTOR
# ---------------------------------------------------------
with tab1:
    st.subheader("Patient Complete Blood Count (CBC)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        mcv = st.number_input("MCV (fL)", min_value=30.0, max_value=130.0, value=68.7, step=0.1)
        hb = st.number_input("Hemoglobin - Hb (g/dL)", min_value=3.0, max_value=20.0, value=10.8, step=0.1)
    with col2:
        mch = st.number_input("MCH (pg)", min_value=10.0, max_value=40.0, value=21.2, step=0.1)
        rdw = st.number_input("RDW (%)", min_value=8.0, max_value=30.0, value=13.4, step=0.1)
    with col3:
        rbc = st.number_input("RBC Count (10^6/µL)", min_value=1.0, max_value=10.0, value=5.12, step=0.01)

    if st.button("Run Diagnostic Inference", type="primary", use_container_width=True):
        patient_raw = pd.DataFrame([{'mcv': mcv, 'mch': mch, 'rbc': rbc, 'hb': hb, 'rdw': rdw}])
        patient_eng = engineer_features(patient_raw)
        
        pred_idx = model.predict(patient_eng)[0]
        probs = model.predict_proba(patient_eng)[0]
        pred_label = le.inverse_transform([pred_idx])[0].upper()

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("Predicted Diagnosis")
            if "ALPHA" in pred_label:
                st.error(f"### {pred_label}")
            elif "BETA" in pred_label:
                st.warning(f"### {pred_label}")
            else:
                st.success(f"### {pred_label}")
            st.caption("Decision generated via Random Forest trained on composite clinical indices.")

        with res_col2:
            prob_df = pd.DataFrame({
                'Phenotype': [c.upper() for c in le.classes_],
                'Probability (%)': probs * 100
            })
            fig_prob = px.bar(
                prob_df, x='Probability (%)', y='Phenotype', orientation='h',
                color='Phenotype', text_auto='.1f',
                title="Model Classification Confidence",
                color_discrete_map={'ALPHA CARRIER': '#EF553B', 'BETA CARRIER': '#FECB52', 'NORMAL': '#00CC96'}
            )
            fig_prob.update_layout(showlegend=False, xaxis_range=[0, 100], height=240, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_prob, use_container_width=True)

        # Derived Clinical Metrics
        st.subheader("Derived Hematological Indices")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Mentzer", f"{patient_eng['mentzer_index'][0]:.2f}", "< 13 Trait")
        m2.metric("Green & King", f"{patient_eng['green_king_index'][0]:.2f}", "< 65 Trait")
        m3.metric("Shine & Lal", f"{patient_eng['shine_lal_index'][0]:.2f}", "< 1530 Trait")
        m4.metric("Srivastava", f"{patient_eng['srivastava_index'][0]:.2f}", "< 3.8 Trait")
        m5.metric("Ehsani", f"{patient_eng['ehsani_index'][0]:.2f}", "< 15 Trait")
        m6.metric("England & Fraser", f"{patient_eng['england_fraser_index'][0]:.2f}", "< 0 Trait")

# ---------------------------------------------------------
# TAB 2: BATCH CSV PROCESSING
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch Dataset Inference")
    st.write("Upload a CSV file with patient records containing columns: `mcv`, `mch`, `rbc`, `hb`, `rdw`")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.strip().str.lower()
        
        req_cols = ['mcv', 'mch', 'rbc', 'hb', 'rdw']
        if all(col in batch_df.columns for col in req_cols):
            batch_eng = engineer_features(batch_df[req_cols])
            preds = model.predict(batch_eng)
            batch_df['predicted_phenotype'] = le.inverse_transform(preds)
            
            st.success(f"Successfully evaluated {len(batch_df)} records!")
            st.dataframe(batch_df, use_container_width=True)
            
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                data=csv,
                file_name="thalassemia_batch_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Missing required columns! Required: {req_cols}")

# ---------------------------------------------------------
# TAB 3: MODEL INTERPRETABILITY
# ---------------------------------------------------------
with tab3:
    st.subheader("Feature Importance Analysis")
    st.write("Gini Impurity Reduction showing the diagnostic weight of each engineered index.")
    
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
    
    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation='h',
        labels={'x': 'Importance Score', 'y': 'Feature / Index'},
        title="Random Forest Relative Feature Contributions",
        color=importances.values,
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(height=480)
    st.plotly_chart(fig_imp, use_container_width=True)
