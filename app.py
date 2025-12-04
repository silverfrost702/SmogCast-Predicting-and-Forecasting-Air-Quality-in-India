import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
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
FORECAST_HORIZON = 7
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
CAT_FEATURES = ['City']
AQI_LABELS = ['Good/Satis', 'Moderate', 'Poor', 'Very Poor/Severe']
AQI_BREAKS = {
    'PM2.5': [(0.0, 30.0, 0, 50), (30.1, 60.0, 51, 100), (60.1, 90.0, 101, 200), (90.1, 120.0, 201, 300), (120.1, 250.0, 301, 400), (250.1, 500.0, 401, 500)],
    'PM10': [(0.0, 50.0, 0, 50), (50.1, 100.0, 51, 100), (100.1, 250.0, 101, 200), (250.1, 350.0, 201, 300), (350.1, 430.0, 301, 400), (430.1, 1000.0, 401, 500)]
}

# ==========================================
# 1. CUSTOM LAYER & AQI HELPER
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

def get_sub_index(c, bph, bpl, iph, ipl):
    return ((iph - ipl) / (bph - bpl)) * (c - bpl) + ipl

def calculate_full_aqi(row):
    sub_indices = []
    pm25_val = max(0.01, row['PM2.5'])
    for bpl, bph, ipl, iph in AQI_BREAKS['PM2.5']:
        if bpl <= pm25_val <= bph: sub_indices.append(get_sub_index(pm25_val, bph, bpl, iph, ipl)); break
    else: sub_indices.append(500.0)
    pm10_val = max(0.01, row['PM10'])
    for bpl, bph, ipl, iph in AQI_BREAKS['PM10']:
        if bpl <= pm10_val <= bph: sub_indices.append(get_sub_index(pm10_val, bph, bpl, iph, ipl)); break
    else: sub_indices.append(500.0)
    return max(sub_indices) if sub_indices else 0.0

# ==========================================
# 2. LOAD MODELS
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        cb_model = CatBoostClassifier()
        cb_model.load_model("catboost_model.cbm")
        custom_objs = get_custom_objects()
        lstm_model = tf.keras.models.load_model("lstm_model.keras", custom_objects=custom_objs)
        trans_model = tf.keras.models.load_model("transformer_model.keras", custom_objects=custom_objs)
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        return cb_model, lstm_model, trans_model, scaler, encoders
    except Exception as e:
        return None, None, None, None, str(e)

# ==========================================
# 3. METRIC CALCULATION HELPERS
# ==========================================
@st.cache_data
def calculate_global_accuracy(_model, df, _encoders):
    # Prepare minimal data for global accuracy check
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['AQI_Calculated'] = df.apply(calculate_full_aqi, axis=1)
    
    # Simple median fill + Safety Fill
    for col in POLLUTANTS + ['AQI_Calculated']:
        df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df.fillna(0, inplace=True)

    def aqi_to_category(aqi):
        if aqi <= 100: return 0 
        elif aqi <= 200: return 1 
        elif aqi <= 300: return 2 
        else: return 3 
    
    df['AQI_Target'] = df['AQI_Calculated'].apply(aqi_to_category)
    
    # Group by City/Date
    df['Date'] = df['Datetime'].dt.date
    daily = df.groupby(['City', 'Date']).agg({
        **{p: 'mean' for p in POLLUTANTS},
        'AQI_Target': lambda x: x.value_counts().index[0]
    }).reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek

    # Feature Engineering (Fast Version)
    for p in POLLUTANTS:
        daily[f'{p}_lag1'] = daily.groupby('City')[p].shift(1)
        daily[f'{p}_roll7'] = daily.groupby('City')[p].transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    
    daily.dropna(inplace=True)
    
    # Sample 20% for testing speed
    _, test_df = train_test_split(daily, test_size=0.2, random_state=42, stratify=daily['AQI_Target'])
    
    features = []
    for p in POLLUTANTS: features.extend([p, f'{p}_lag1', f'{p}_roll7'])
    features.extend(['City', 'DayOfWeek'])
    
    X = test_df[features].copy()
    y = test_df['AQI_Target'].values
    
    # Encode
    for col in ['City', 'DayOfWeek']:
        if col in _encoders:
            X[col] = _encoders[col].transform(X[col])
            
    y_pred = _model.predict(X)
    return accuracy_score(y, y_pred)

def calculate_city_metrics(city, df, _lstm, _trans, _scaler):
    city_df = df[df['City'] == city].copy()
    city_df['Datetime'] = pd.to_datetime(city_df['Datetime'])
    city_df = city_df.sort_values('Datetime')
    
    # Fill
    for col in POLLUTANTS:
        city_df[col] = city_df[col].fillna(city_df[col].median())
    city_df.fillna(0, inplace=True)
    
    # Daily
    city_df['Date'] = city_df['Datetime'].dt.date
    daily = city_df.groupby('Date')[POLLUTANTS].mean().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    
    if len(daily) < SEQUENCE_LENGTH + FORECAST_HORIZON:
        return None, None
        
    # Temporal Features
    daily['DayOfYear'] = daily['Date'].dt.dayofyear
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek
    daily['sin_doy'] = np.sin(2 * np.pi * daily['DayOfYear'] / 365)
    daily['cos_doy'] = np.cos(2 * np.pi * daily['DayOfYear'] / 365)
    daily['sin_dow'] = np.sin(2 * np.pi * daily['DayOfWeek'] / 7)
    daily['cos_dow'] = np.cos(2 * np.pi * daily['DayOfWeek'] / 7)
    
    temp_feats = ['sin_doy', 'cos_doy', 'sin_dow', 'cos_dow']
    data = daily[POLLUTANTS + temp_feats].values
    data[:, 0:12] = np.log1p(data[:, 0:12])
    data_scaled = _scaler.transform(data)
    
    # Create Test Sequences (Last 10% of data)
    test_size = int(len(data_scaled) * 0.1)
    if test_size < 1: return None, None
    
    X, Y = [], []
    # Start loop from end - test_size
    start_idx = len(data_scaled) - test_size - SEQUENCE_LENGTH - FORECAST_HORIZON
    if start_idx < 0: start_idx = 0
    
    for i in range(start_idx, len(data_scaled) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1):
        X.append(data_scaled[i : i + SEQUENCE_LENGTH])
        Y.append(data_scaled[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + FORECAST_HORIZON, 0:2])
        
    X = np.array(X)
    Y = np.array(Y)
    
    if len(X) == 0: return None, None
    
    # Predict
    lstm_pred = _lstm.predict(X, verbose=0)
    trans_pred = _trans.predict(X, verbose=0)
    
    # Helper for inverse (handling 16 features shape)
    def get_actuals(preds):
        dummy = np.zeros((preds.shape[0] * preds.shape[1], 16))
        dummy[:, 0:2] = preds.reshape(-1, 2)
        inv = _scaler.inverse_transform(dummy)[:, 0:2]
        return np.expm1(inv)

    y_true_flat = get_actuals(Y)
    lstm_flat = get_actuals(lstm_pred)
    trans_flat = get_actuals(trans_pred)
    
    # Metrics
    lstm_mae = mean_absolute_error(y_true_flat, lstm_flat)
    lstm_rmse = np.sqrt(mean_squared_error(y_true_flat, lstm_flat))
    
    trans_mae = mean_absolute_error(y_true_flat, trans_flat)
    trans_rmse = np.sqrt(mean_squared_error(y_true_flat, trans_flat))
    
    return (lstm_mae, lstm_rmse), (trans_mae, trans_rmse)

# ==========================================
# 4. PREDICTION HELPERS
# ==========================================
def prep_data_for_prediction(df, city):
    city_df = df[df['City'] == city].copy()
    city_df['Datetime'] = pd.to_datetime(city_df['Datetime'])
    city_df = city_df.sort_values('Datetime')
    for col in POLLUTANTS: city_df[col] = city_df[col].fillna(city_df[col].median())
    city_df.fillna(0, inplace=True)
    city_df['Date'] = city_df['Datetime'].dt.date
    daily = city_df.groupby('Date')[POLLUTANTS].mean().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    return daily

def get_catboost_features(daily_df, encoders, city):
    if len(daily_df) < 8: return None
    last_row = daily_df.iloc[[-1]].copy()
    features = {}
    for p in POLLUTANTS:
        features[p] = last_row[p].values[0]
        features[f'{p}_lag1'] = daily_df.iloc[-2][p]
        features[f'{p}_roll7'] = daily_df.iloc[-8:-1][p].mean()
    features['City'] = city
    features['DayOfWeek'] = last_row['Date'].dt.dayofweek.values[0]
    input_df = pd.DataFrame([features])
    for col in ['City', 'DayOfWeek']:
        if col in encoders:
            le = encoders[col]
            val = input_df[col].iloc[0]
            input_df[col] = le.transform([val]) if val in le.classes_ else 0
    
    feat_order = []
    for p in POLLUTANTS: feat_order.extend([p, f'{p}_lag1', f'{p}_roll7'])
    feat_order.extend(['City', 'DayOfWeek'])
    return input_df[feat_order]

def get_dl_sequence(daily_df, scaler):
    if len(daily_df) < SEQUENCE_LENGTH: return None
    seq_df = daily_df.iloc[-SEQUENCE_LENGTH:].copy()
    seq_df['DayOfYear'] = seq_df['Date'].dt.dayofyear
    seq_df['DayOfWeek'] = seq_df['Date'].dt.dayofweek
    seq_df['sin_doy'] = np.sin(2 * np.pi * seq_df['DayOfYear'] / 365)
    seq_df['cos_doy'] = np.cos(2 * np.pi * seq_df['DayOfYear'] / 365)
    seq_df['sin_dow'] = np.sin(2 * np.pi * seq_df['DayOfWeek'] / 7)
    seq_df['cos_dow'] = np.cos(2 * np.pi * seq_df['DayOfWeek'] / 7)
    data = seq_df[POLLUTANTS + ['sin_doy', 'cos_doy', 'sin_dow', 'cos_dow']].values
    data[:, 0:12] = np.log1p(data[:, 0:12])
    data_scaled = scaler.transform(data)
    return np.expand_dims(data_scaled, axis=0)

# ==========================================
# 5. MAIN APP UI
# ==========================================
st.title("⚡ SmogCast: Instant Air Quality AI")
st.markdown("### Pre-trained Models: CatBoost (classification) | LSTM & Transformer (Forecast)")

# Load Everything
with st.spinner("Loading AI Brains..."):
    cb_model, lstm_model, trans_model, scaler, encoders = load_artifacts()

if isinstance(encoders, str):
    st.error(f"Error loading models: {encoders}")
    st.stop()

# Load Data
raw_df = pd.read_csv(FILE_PATH)
cities = sorted(raw_df['City'].unique())

# Controls
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Controls")
    selected_city = st.selectbox("Select City", cities)
    run_btn = st.button("Generate Forecast", type="primary")
    
    # --- METRICS SECTION IN SIDEBAR ---
    st.markdown("---")
    st.markdown("### Model Performance")
    
    # 1. Global Accuracy
    if 'global_acc' not in st.session_state:
        with st.spinner("Calculating Accuracy..."):
            st.session_state['global_acc'] = calculate_global_accuracy(cb_model, raw_df.copy(), encoders)
    
    st.metric("Global Classification Accuracy", f"{st.session_state['global_acc']*100:.2f}%")

with col2:
    if run_btn:
        daily_data = prep_data_for_prediction(raw_df, selected_city)
        
        # --- CALCULATE LOCAL METRICS ---
        with st.spinner(f"Evaluating model on {selected_city} data..."):
            lstm_metrics, trans_metrics = calculate_city_metrics(selected_city, raw_df, lstm_model, trans_model, scaler)

        if lstm_metrics:
             m_col1, m_col2 = st.columns(2)
             m_col1.metric(f"LSTM MAE ({selected_city})", f"{lstm_metrics[0]:.2f}", delta_color="inverse")
             m_col1.caption(f"RMSE: {lstm_metrics[1]:.2f}")
             
             m_col2.metric(f"Transformer MAE ({selected_city})", f"{trans_metrics[0]:.2f}", delta_color="inverse")
             m_col2.caption(f"RMSE: {trans_metrics[1]:.2f}")
        
        st.divider()

        if len(daily_data) < SEQUENCE_LENGTH:
            st.warning(f"Not enough historical data for {selected_city}.")
        else:
            # 1. Classification
            cat_input = get_catboost_features(daily_data, encoders, selected_city)
            if cat_input is not None:
                pred_class = cb_model.predict(cat_input)[0][0]
                status = AQI_LABELS[pred_class]
                colors = {'Good/Satis': '#00e400', 'Moderate': '#ffff00', 'Poor': '#ff7e00', 'Very Poor/Severe': '#ff0000'}
                c = colors.get(status, '#ffffff')
                st.markdown(f"""
                <div style="padding: 20px; background-color: {c}; border-radius: 10px; color: black; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin:0;">Current Status: {status}</h2>
                    <p style="margin:0;">Based on latest available data pattern</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 2. Forecasting
            seq_input = get_dl_sequence(daily_data, scaler)
            lstm_pred_scaled = lstm_model.predict(seq_input, verbose=0)
            trans_pred_scaled = trans_model.predict(seq_input, verbose=0)
            
            def inverse_predictions(pred_scaled):
                dummy = np.zeros((7, 16))
                dummy[:, 0:2] = pred_scaled[0]
                inverse = scaler.inverse_transform(dummy)
                return np.expm1(inverse[:, 0:2])
            
            lstm_final = inverse_predictions(lstm_pred_scaled)
            trans_final = inverse_predictions(trans_pred_scaled)
            
            days = [f"Day +{i+1}" for i in range(7)]
            
            fig_pm25 = go.Figure()
            fig_pm25.add_trace(go.Scatter(x=days, y=lstm_final[:, 0], mode='lines+markers', name='Bi-LSTM', line=dict(color='red', width=2)))
            fig_pm25.add_trace(go.Scatter(x=days, y=trans_final[:, 0], mode='lines+markers', name='Transformer', line=dict(color='green', width=2, dash='dot')))
            fig_pm25.update_layout(title=f"PM2.5 Forecast", yaxis_title="µg/m³", template="plotly_dark")
            st.plotly_chart(fig_pm25, use_container_width=True)
            
            fig_pm10 = go.Figure()
            fig_pm10.add_trace(go.Scatter(x=days, y=lstm_final[:, 1], mode='lines+markers', name='Bi-LSTM', line=dict(color='red', width=2)))
            fig_pm10.add_trace(go.Scatter(x=days, y=trans_final[:, 1], mode='lines+markers', name='Transformer', line=dict(color='green', width=2, dash='dot')))
            fig_pm10.update_layout(title=f"PM10 Forecast", yaxis_title="µg/m³", template="plotly_dark")
            st.plotly_chart(fig_pm10, use_container_width=True)
    else:
        st.info("👈 Select a city and click 'Generate Forecast'")