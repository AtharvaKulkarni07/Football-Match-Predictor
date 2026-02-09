import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="⚽ Football Predictor",
    page_icon="⚽",
    layout="centered"
)

# Title and description
st.title("⚽ Football Match Predictor")
st.markdown("Predict if the home team will win using machine learning")

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        models = {
            'lr': pickle.load(open('models/logistic_regression_model.pkl', 'rb')),
            'rf': pickle.load(open('models/random_forest_model.pkl', 'rb')),
            'gb': pickle.load(open('models/gradient_boosting_model.pkl', 'rb'))
        }
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        feature_names = pickle.load(open('models/feature_names.pkl', 'rb'))
        return models, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

models, scaler, feature_names = load_models()

if models is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select", ["Home", "Prediction", "About"])

# ============== HOME PAGE ==============
if page == "Home":
    st.header("Welcome!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Models", "3", "LR, RF, GB")
        st.metric("Avg Accuracy", "63.3%", "Train Data")
    
    with col2:
        st.metric("Best ROC AUC", "0.69", "Gradient Boosting")
        st.metric("Training Matches", "5,472", "Historic Data")
    
    st.markdown("---")
    st.subheader("📊 How to Use")
    st.write("""
    1. Go to the **Prediction** tab
    2. Enter home team and away team statistics
    3. Get an instant prediction with confidence score
    """)
    
    st.subheader("🎯 Features Used")
    st.write("The model uses 18 features including:")
    st.write("• Goals scored & conceded • Team points • Recent form • Goal differences")

# ============== PREDICTION PAGE ==============
elif page == "Prediction":
    st.header("Make a Prediction")
    
    st.markdown("### Enter Match Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        htgs = st.number_input("Goals Scored", min_value=0, max_value=10, value=2, key="htgs")
        htgc = st.number_input("Goals Conceded", min_value=0, max_value=10, value=1, key="htgc")
        htp = st.number_input("Points", min_value=0, max_value=100, value=30, key="htp")
        htm = st.number_input("Matches Played", min_value=1, max_value=40, value=15, key="htm")
    
    with col2:
        st.subheader("⚽ Away Team")
        atgs = st.number_input("Goals Scored", min_value=0, max_value=10, value=1, key="atgs")
        atgc = st.number_input("Goals Conceded", min_value=0, max_value=10, value=2, key="atgc")
        atp = st.number_input("Points", min_value=0, max_value=100, value=25, key="atp")
        atm = st.number_input("Matches Played", min_value=1, max_value=40, value=15, key="atm")
    
    # Create prediction button
    if st.button("🔮 Predict Result", use_container_width=True, type="primary"):
        # Prepare features
        features = np.array([[htgs, htgc, htp, htm, atgs, atgc, atp, atm, 
                             htgs - atgs, htp - atp, htgs / htm, atgs / atm, 
                             htgc / htm, atgc / atm, htp / htm, atp / atm, 
                             (htgs - htgc) / htm, (atgs - atgc) / atm]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions
        lr_pred = models['lr'].predict_proba(features_scaled)[0][1]
        rf_pred = models['rf'].predict_proba(features_scaled)[0][1]
        gb_pred = models['gb'].predict_proba(features_scaled)[0][1]
        
        # Ensemble average
        ensemble_pred = (lr_pred + rf_pred + gb_pred) / 3
        ensemble_class = "Home Win 🎉" if ensemble_pred > 0.5 else "Home Draw/Loss ⚠️"
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Logistic Regression", f"{lr_pred*100:.1f}%")
        with col2:
            st.metric("Random Forest", f"{rf_pred*100:.1f}%")
        with col3:
            st.metric("Gradient Boosting", f"{gb_pred*100:.1f}%")
        
        st.markdown("---")
        
        # Ensemble prediction
        color = "green" if ensemble_pred > 0.6 else "orange" if ensemble_pred > 0.4 else "red"
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">Ensemble Prediction</h2>
            <h3 style="color: white; margin: 10px 0 0 0;">{ensemble_class}</h3>
            <p style="color: white; font-size: 18px; margin: 10px 0 0 0;">Confidence: {ensemble_pred*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Tip: Predictions are based on historical data. Multiple factors can influence real match outcomes.")

# ============== ABOUT PAGE ==============
elif page == "About":
    st.header("About This Project")
    
    st.subheader("🎯 Project Overview")
    st.write("""
    This football match predictor uses machine learning to forecast home team victories.
    The model analyzes 18 statistical features from 6,840 historic football matches.
    """)
    
    st.subheader("🤖 Models Used")
    st.write("""
    - **Logistic Regression**: Fast, interpretable baseline model
    - **Random Forest**: Captures complex non-linear patterns
    - **Gradient Boosting**: Best ROC AUC (0.69) sequential learning approach
    - **Ensemble**: Combines all three models for robust predictions
    """)
    
    st.subheader("📊 Dataset Information")
    st.write("""
    - Total Matches: 6,840
    - Training Set: 5,472 matches (80%)
    - Test Set: 1,368 matches (20%)
    - Historical Home Win Rate: 46.43%
    """)
    
    st.subheader("⚠️ Disclaimer")
    st.write("""
    This predictor is for educational purposes. Real match outcomes depend on many 
    unpredictable factors (injuries, weather, motivation, etc.) that models cannot capture.
    """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>⚽ Football Match Prediction Dashboard v1.0</p>", 
                unsafe_allow_html=True)
