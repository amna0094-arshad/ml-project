import streamlit as st
import joblib
import numpy as np

# Page title
st.set_page_config(page_title="Fetal Health Predictor")
st.title("Fetal Health Prediction App")
st.write("Enter CTG features to predict fetal health state.")

# Load model and scaler
model = joblib.load('rf_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Input fields
LB = st.slider('LB - FHR Baseline', 50, 200, 120)
AC = st.slider('AC - Accelerations', 0, 20, 0)
FM = st.slider('FM - Fetal Movements', 0, 600, 0)
UC = st.slider('UC - Uterine Contractions', 0, 20, 0)
ASTV = st.slider('ASTV', 0, 100, 73)
MSTV = st.slider('MSTV', 0.0, 10.0, 0.5)
ALTV = st.slider('ALTV', 0, 100, 43)
MLTV = st.slider('MLTV', 0.0, 50.0, 2.4)
DL = st.slider('DL', 0, 20, 0)
DS = st.slider('DS', 0, 10, 0)
DP = st.slider('DP', 0, 10, 0)
DR = st.slider('DR', 0, 10, 0)
Width = st.slider('Width', 0, 300, 64)
Min = st.slider('Min', 50, 200, 62)
Max = st.slider('Max', 100, 300, 126)
Nmax = st.slider('Nmax', 0, 20, 2)
Nzeros = st.slider('Nzeros', 0, 10, 0)
Mode = st.slider('Mode', 50, 200, 120)
Mean = st.slider('Mean', 50, 200, 137)
Median = st.slider('Median', 50, 200, 121)
Variance = st.slider('Variance', 0, 300, 73)
Tendency = st.slider('Tendency', -1, 1, 1)

# Predict button
if st.button('Predict'):
    features = np.array([[LB, AC, FM, UC, ASTV, MSTV, ALTV, MLTV,
                          DL, DS, DP, DR, Width, Min, Max, Nmax,
                          Nzeros, Mode, Mean, Median, Variance, Tendency]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    fetal_state = {1.0: '✅ Normal', 2.0: '⚠️ Suspect', 3.0: '🚨 Pathologic'}
    result = fetal_state[prediction[0]]
    
    st.success(f"Predicted Fetal State: {result}")