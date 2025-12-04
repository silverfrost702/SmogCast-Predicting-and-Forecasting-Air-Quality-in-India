import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from catboost import CatBoostClassifier
import joblib
import plotly.graph_objects as go
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Air Quality AI (Instant)", layout="wide")

# --- Constants ---
FILE_PATH = 'city_hour.csv'
SEQUENCE_LENGTH = 45
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
CAT_FEATURES = ['City']
AQI_LABELS = ['Good/Satis', 'Moderate', 'Poor', 'Very Poor/Severe']

# ==========================================
# 1. CUSTOM LAYER DEFINITION (Required for Loading)
# ==========================================
@st.cache_resource
def get_custom_objects():
    class PositionalEncoding(Layer):
        def __init__(self, sequence_length=500, d_model=128, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.d_model = d_model
            pe = np.zeros((sequence_length, d_model))
            position = np.arange(0, sequence_length, dtype=np.float32)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        def call(self, inputs):
            return inputs + self.pe[:, :tf.shape(inputs)[1]]
        def get_config(self):
            config = super().get_config()
            config.update({"sequence_length": 500, "d_model": self.d_model})
            return config
    return {"PositionalEncoding": PositionalEncoding}

# ==========================================
# 2. LOAD MODELS (Cached = Instant)
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        # Load CatBoost
        cb_model = CatBoostClassifier()
        cb_model.load_model("catboost_model.cbm")
        
        # Load Keras Models with Custom Layer
        custom_objs = get_custom_objects()
        lstm_model = tf.keras.models.load_model("lstm_model.keras", custom_objects=custom_objs)
        trans_model = tf.keras.models.load_model("transformer_model.keras", custom_objects=custom_objs)
        
        # Load Scalers
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        
        return cb_model, lstm_model, trans_model, scaler, encoders
    except Exception as e:
        return None, None, None, None, str(e)

# ==========================================
# 3. DATA PREPARATION HELPERS
# ==========================================
def prep_data_for_prediction(df, city):
    # Filter City
    city_df = df[df['City'] == city].copy()
    city_df['Datetime'] = pd.to_datetime(city_df['Datetime'])
    city_df = city_df.sort_values('Datetime')
    
    # Impute Missing Values (Same logic as training)
    for col in POLLUTANTS:
        city_df[col] = city_df[col].fillna(city_df[col].median())
    city_df.fillna(0, inplace=True) # Safety fill
    
    # Daily Aggregation
    city_df['Date'] = city_df['Datetime'].dt.date
    daily = city_df.groupby('Date')[POLLUTANTS].mean().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    
    return daily

def get_catboost_features(daily_df, encoders, city):
    # Need Lag Features for the LATEST day
    if len(daily_df) < 8: return None # Need at least 8 days for 7-day rolling
    
    last_row = daily_df.iloc[[-1]].copy()
    
    # Create Features manually for the last row
    # (In a real app, you'd calculate these rolling stats properly over the whole series)
    features = {}
    for p in POLLUTANTS:
        features[p] = last_row[p].values[0]
        # Approximation: Using previous day as lag1
        features[f'{p}_lag1'] = daily_df.iloc[-2][p]
        # Approximation: Using last 7 days mean
        features[f'{p}_roll7'] = daily_df.iloc[-8:-1][p].mean()
    
    features['City'] = city
    features['DayOfWeek'] = last_row['Date'].dt.dayofweek.values[0]
    
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    
    # Encode
    for col in ['City', 'DayOfWeek']:
        if col in encoders:
            le = encoders[col]
            # Handle unknown labels safely
            val = input_df[col].iloc[0]
            if val in le.classes_:
                input_df[col] = le.transform([val])
            else:
                input_df[col] = 0 # Default to 0 if unknown
                
    # Ensure column order matches training
    # (POLLUTANTS + Lag Features + City + DayOfWeek)
    feature_order = []
    for p in POLLUTANTS:
        feature_order.extend([p, f'{p}_lag1', f'{p}_roll7'])
    feature_order.extend(['City', 'DayOfWeek'])
    
    return input_df[feature_order]

def get_dl_sequence(daily_df, scaler):
    if len(daily_df) < SEQUENCE_LENGTH: return None
    
    # Take last 45 days
    seq_df = daily_df.iloc[-SEQUENCE_LENGTH:].copy()
    
    # Add Temporal Features
    seq_df['DayOfYear'] = seq_df['Date'].dt.dayofyear
    seq_df['DayOfWeek'] = seq_df['Date'].dt.dayofweek
    
    seq_df['sin_doy'] = np.sin(2 * np.pi * seq_df['DayOfYear'] / 365)
    seq_df['cos_doy'] = np.cos(2 * np.pi * seq_df['DayOfYear'] / 365)
    seq_df['sin_dow'] = np.sin(2 * np.pi * seq_df['DayOfWeek'] / 7)
    seq_df['cos_dow'] = np.cos(2 * np.pi * seq_df['DayOfWeek'] / 7)
    
    temp_feats = ['sin_doy', 'cos_doy', 'sin_dow', 'cos_dow']
    data = seq_df[POLLUTANTS + temp_feats].values
    
    # Log Transform
    data[:, 0:12] = np.log1p(data[:, 0:12])
    
    # Scale
    data_scaled = scaler.transform(data)
    
    # Reshape for Model (1, 45, 16)
    return np.expand_dims(data_scaled, axis=0)

# ==========================================
# 4. MAIN APP UI
# ==========================================

st.title("⚡ SmogCast: Instant Air Quality AI")
st.markdown("### Pre-trained Models: CatBoost (Status) | LSTM & Transformer (Forecast)")

# 1. Load Everything
with st.spinner("Loading AI Brains..."):
    cb_model, lstm_model, trans_model, scaler, encoders = load_artifacts()

if isinstance(encoders, str): # Error handling
    st.error(f"Failed to load models. Please ensure .keras/.cbm/.pkl files are in GitHub root.\nError: {encoders}")
    st.stop()

# 2. Load Data for Selection
try:
    raw_df = pd.read_csv(FILE_PATH)
    cities = sorted(raw_df['City'].unique())
except:
    st.error("Could not load city_hour.csv")
    st.stop()

# 3. UI Controls
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Controls")
    selected_city = st.selectbox("Select City", cities)
    run_btn = st.button("Generate Forecast", type="primary")

with col2:
    if run_btn:
        # --- A. PREPARE DATA ---
        daily_data = prep_data_for_prediction(raw_df, selected_city)
        
        if len(daily_data) < SEQUENCE_LENGTH:
            st.warning(f"Not enough historical data for {selected_city} to generate a forecast.")
        else:
            # --- B. CLASSIFICATION (CatBoost) ---
            cat_input = get_catboost_features(daily_data, encoders, selected_city)
            if cat_input is not None:
                pred_class = cb_model.predict(cat_input)[0][0]
                status = AQI_LABELS[pred_class]
                
                # Dynamic Color
                colors = {'Good/Satis': '#00e400', 'Moderate': '#ffff00', 'Poor': '#ff7e00', 'Very Poor/Severe': '#ff0000'}
                c = colors.get(status, '#ffffff')
                
                st.markdown(f"""
                <div style="padding: 20px; background-color: {c}; border-radius: 10px; color: black; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin:0;">Current Status: {status}</h2>
                    <p style="margin:0;">Based on latest available data pattern</p>
                </div>
                """, unsafe_allow_html=True)
            
            # --- C. FORECASTING (Deep Learning) ---
            seq_input = get_dl_sequence(daily_data, scaler)
            
            # Predict
            lstm_pred_scaled = lstm_model.predict(seq_input, verbose=0)
            trans_pred_scaled = trans_model.predict(seq_input, verbose=0)
            
            # Inverse Transform Logic
            # We need to construct a dummy array matching the scaler's shape (16 features)
            # The model predicts only 2 features (PM2.5, PM10). We fill the rest with zeros to inverse.
            def inverse_predictions(pred_scaled):
                dummy = np.zeros((7, 16))
                dummy[:, 0:2] = pred_scaled[0] # Fill first 2 cols
                inverse = scaler.inverse_transform(dummy)
                return np.expm1(inverse[:, 0:2]) # Expm1 to reverse Log1p
            
            lstm_final = inverse_predictions(lstm_pred_scaled)
            trans_final = inverse_predictions(trans_pred_scaled)
            
            # --- D. VISUALIZATION ---
            st.subheader("7-Day Forecast (PM2.5 & PM10)")
            
            days = [f"Day +{i+1}" for i in range(7)]
            
            # PM2.5 Chart
            fig_pm25 = go.Figure()
            fig_pm25.add_trace(go.Scatter(x=days, y=lstm_final[:, 0], mode='lines+markers', name='Bi-LSTM', line=dict(color='red', width=2)))
            fig_pm25.add_trace(go.Scatter(x=days, y=trans_final[:, 0], mode='lines+markers', name='Transformer', line=dict(color='green', width=2, dash='dot')))
            fig_pm25.update_layout(title=f"PM2.5 Forecast for {selected_city}", yaxis_title="µg/m³", template="plotly_dark")
            st.plotly_chart(fig_pm25, use_container_width=True)
            
            # PM10 Chart
            fig_pm10 = go.Figure()
            fig_pm10.add_trace(go.Scatter(x=days, y=lstm_final[:, 1], mode='lines+markers', name='Bi-LSTM', line=dict(color='red', width=2)))
            fig_pm10.add_trace(go.Scatter(x=days, y=trans_final[:, 1], mode='lines+markers', name='Transformer', line=dict(color='green', width=2, dash='dot')))
            fig_pm10.update_layout(title=f"PM10 Forecast for {selected_city}", yaxis_title="µg/m³", template="plotly_dark")
            st.plotly_chart(fig_pm10, use_container_width=True)

    else:
        st.info("👈 Select a city and click 'Generate Forecast'")