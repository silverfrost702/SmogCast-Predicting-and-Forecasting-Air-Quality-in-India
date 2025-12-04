import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer, MultiHeadAttention, Normalization, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import warnings
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Streamlit Page Config ---
st.set_page_config(page_title="Air Quality Forecasting AI", layout="wide")
st.write("**v4.0 - Fast & Fixed (Epochs=1, Grouping Fixed)**") 

# --- Configuration Constants ---
FILE_PATH = 'city_hour.csv' 
FORECAST_HORIZON = 7 
SEQUENCE_LENGTH = 45 
BATCH_SIZE = 32
# SPEED UPDATE: Epochs set to 1 for maximum speed as requested
EPOCHS = 15 
RANDOM_SEED = 42
POLLUTANTS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
CAT_FEATURES = ['City'] 
AQI_LABELS_FULL = ['Good/Satis', 'Moderate', 'Poor', 'Very Poor/Severe'] 

# ==============================================================================
# Helper Classes and Functions
# ==============================================================================

class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
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

# AQI Calculation Logic
AQI_BREAKS = {
    'PM2.5': [(0.0, 30.0, 0, 50), (30.1, 60.0, 51, 100), (60.1, 90.0, 101, 200), (90.1, 120.0, 201, 300), (120.1, 250.0, 301, 400), (250.1, 500.0, 401, 500)],
    'PM10': [(0.0, 50.0, 0, 50), (50.1, 100.0, 51, 100), (100.1, 250.0, 101, 200), (250.1, 350.0, 201, 300), (350.1, 430.0, 301, 400), (430.1, 1000.0, 401, 500)]
}

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

# ==============================================================================
# Cached Data Processing Functions (FIXED)
# ==============================================================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def prepare_classification_data(df):
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['AQI_Calculated'] = df.apply(calculate_full_aqi, axis=1)
    
    # 1. Fill with median
    for col in POLLUTANTS + ['AQI_Calculated']:
        df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    
    # 2. Safety Fill
    df.fillna(0, inplace=True) 
    
    def aqi_to_category(aqi):
        if aqi <= 100: return 0 
        elif aqi <= 200: return 1 
        elif aqi <= 300: return 2 
        else: return 3 
        
    df['AQI_Target'] = df['AQI_Calculated'].apply(aqi_to_category)
    
    # --- MAJOR FIX: Group by City AND Date ---
    # This prevents Delhi data from being merged with other cities
    df['Date'] = df['Datetime'].dt.date
    daily_df = df.groupby(['City', 'Date']).agg({
        **{p: 'mean' for p in POLLUTANTS},
        'AQI_Target': lambda x: x.value_counts().index[0]
    }).reset_index()
    
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    
    lag_features = []
    for p in POLLUTANTS:
        daily_df[f'{p}_lag1'] = daily_df.groupby('City')[p].shift(1)
        lag_features.append(f'{p}_lag1')
        daily_df[f'{p}_roll7'] = daily_df.groupby('City')[p].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean().shift(1)
        )
        lag_features.append(f'{p}_roll7')
        
    daily_df.dropna(inplace=True)
    return daily_df, lag_features

@st.cache_data
def prepare_forecasting_data(df, target_city):
    city_df = df[df['City'] == target_city].copy()
    city_df['Datetime'] = pd.to_datetime(city_df['Datetime'])
    city_df = city_df.sort_values('Datetime').reset_index(drop=True)
    city_df[POLLUTANTS] = city_df[POLLUTANTS].ffill()
    
    # Safety fill for forecasting
    city_df[POLLUTANTS] = city_df[POLLUTANTS].fillna(0)
    city_df = city_df.dropna(subset=POLLUTANTS)
    
    city_df['DayOfYear'] = city_df['Datetime'].dt.dayofyear
    city_df['DayOfWeek'] = city_df['Datetime'].dt.dayofweek
    city_df['sin_doy'] = np.sin(2 * np.pi * city_df['DayOfYear'] / 365)
    city_df['cos_doy'] = np.cos(2 * np.pi * city_df['DayOfYear'] / 365)
    city_df['sin_dow'] = np.sin(2 * np.pi * city_df['DayOfWeek'] / 7)
    city_df['cos_dow'] = np.cos(2 * np.pi * city_df['DayOfWeek'] / 7)
    
    TEMPORAL_FEATURES = ['sin_doy', 'cos_doy', 'sin_dow', 'cos_dow']
    forecast_features = POLLUTANTS + TEMPORAL_FEATURES
    data = city_df[forecast_features].values
    
    data[:, 0:12] = np.log1p(data[:, 0:12]) 
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, Y = [], []
    for i in range(len(data_scaled) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1):
        X.append(data_scaled[i : i + SEQUENCE_LENGTH])
        Y.append(data_scaled[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + FORECAST_HORIZON, 0:2])
        
    X = np.array(X)
    Y = np.array(Y)
    
    N = len(X)
    if N == 0: return None, None, None, None, None, None, None
    
    split_train = int(0.8 * N)
    split_val = int(0.9 * N) 

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]
    
    dummy_inverse_scaler = MinMaxScaler()
    dummy_inverse_scaler.min_, dummy_inverse_scaler.scale_ = scaler.min_[0:2], scaler.scale_[0:2]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, dummy_inverse_scaler

# ==============================================================================
# Model Training
# ==============================================================================

@st.cache_resource
def train_classification_model(daily_df, lag_features):
    features = POLLUTANTS + lag_features + CAT_FEATURES + ['DayOfWeek']
    X = daily_df[features].copy()
    y = daily_df['AQI_Target'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp)
    
    cat_cols_to_encode = CAT_FEATURES + ['DayOfWeek']
    X_train_smote = X_train.copy()
    le_encoders = {}
    for col in cat_cols_to_encode:
        le = LabelEncoder()
        X_train_smote[col] = le.fit_transform(X_train_smote[col])
        le_encoders[col] = le
        
    sm = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train_smote, y_train)

    for col in cat_cols_to_encode:
        X_train_res[col] = X_train_res[col].astype('int').map(lambda x: le_encoders[col].inverse_transform([x])[0] if x in le_encoders[col].classes_ else x)

    cat_features_indices = [X_train_res.columns.get_loc(col) for col in cat_cols_to_encode]
    
    cb_model = CatBoostClassifier(
        iterations=200, learning_rate=0.1, loss_function='MultiClass', eval_metric='Accuracy',
        random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False
    )
    
    def prepare_test_set(X_set):
        X_set_encoded = X_set.copy()
        for col in cat_cols_to_encode:
            X_set_encoded[col] = X_set_encoded[col].apply(lambda x: le_encoders[col].transform([x])[0] if x in le_encoders[col].classes_ else np.nan)
            X_set_encoded[col] = X_set_encoded[col].astype(float).map(lambda x: le_encoders[col].inverse_transform([int(x)])[0] if not np.isnan(x) else 'Unknown')
        return X_set_encoded

    cb_model.fit(X_train_res, y_train_res, cat_features=cat_features_indices, early_stopping_rounds=20, eval_set=(prepare_test_set(X_val), y_val))
    
    X_test_final = prepare_test_set(X_test)
    y_pred = cb_model.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred.flatten())
    
    return cb_model, accuracy, le_encoders

@st.cache_resource
def train_lstm(_X_train, _Y_train, _X_val, _Y_val, input_shape):
    model = Sequential([
        Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape), 
        Dropout(0.3),
        Bidirectional(LSTM(units=64)), 
        Dropout(0.3),
        Dense(FORECAST_HORIZON * 2) 
    ])
    model.add(tf.keras.layers.Reshape((FORECAST_HORIZON, 2)))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    model.fit(_X_train, _Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(_X_val, _Y_val), callbacks=callbacks, verbose=0)
    return model

@st.cache_resource
def train_transformer(_X_train, _Y_train, _X_val, _Y_val, input_shape):
    seq_len, d_model = input_shape
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(seq_len, d_model)(inputs)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = attn_output + x
    x = Normalization(axis=-1)(x)
    ffn_output = Dense(256, activation='relu')(x)
    ffn_output = Dense(d_model)(ffn_output)
    x = ffn_output + x
    x = Normalization(axis=-1)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(FORECAST_HORIZON * 2)(x)
    outputs = tf.keras.layers.Reshape((FORECAST_HORIZON, 2))(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    model.fit(_X_train, _Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(_X_val, _Y_val), callbacks=callbacks, verbose=0)
    return model

# ==============================================================================
# UI Implementation
# ==============================================================================

st.title("🏭 Air Quality Forecasting & Classification")
st.markdown("Features: **CatBoost Classifier** (AQI Category) | **Bi-LSTM & Transformer** (PM2.5/PM10 Forecasting)")

# 1. Load Data
df = load_data()

if df is not None:
    cities = sorted(df['City'].unique())
    
    # Sidebar
    st.sidebar.header("Controls")
    selected_city = st.sidebar.selectbox("Select Target City", cities)
    run_btn = st.sidebar.button("Run Models")

    if run_btn:
        st.divider()
        
        # --- PART 1: CLASSIFICATION ---
        st.subheader(f"1. AQI Classification (CatBoost)")
        
        with st.spinner('Training/Loading Classification Model...'):
            daily_df_class, lag_feats = prepare_classification_data(df.copy())
            cb_model, acc, encoders = train_classification_model(daily_df_class, lag_feats)
        
        st.metric("Model Test Accuracy", f"{acc*100:.2f}%")
        
        city_subset = daily_df_class[daily_df_class['City'] == selected_city]
        
        if len(city_subset) > 0:
            city_latest = city_subset.iloc[[-1]]
            
            features = POLLUTANTS + lag_feats + CAT_FEATURES + ['DayOfWeek']
            X_latest = city_latest[features].copy()
            for col in CAT_FEATURES + ['DayOfWeek']:
                val = X_latest[col].values[0]
                if val in encoders[col].classes_:
                    X_latest[col] = encoders[col].transform([val])
                else:
                    X_latest[col] = -1 
            
            pred_class = cb_model.predict(X_latest)[0][0]
            status_text = AQI_LABELS_FULL[pred_class]
            color_map = {'Good/Satis': 'green', 'Moderate': 'orange', 'Poor': 'red', 'Very Poor/Severe': 'purple'}
            
            st.markdown(f"**Current Estimated Status for {selected_city}:** <span style='color:{color_map.get(status_text, 'black')}; font-size:1.2em; font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)
        else:
            st.warning(f"Insufficient data to classify current status for {selected_city} (Global model still accurate).")

        # --- PART 2: FORECASTING ---
        st.divider()
        st.subheader(f"2. 7-Day Forecasting (LSTM vs Transformer)")
        
        with st.spinner(f'Preparing Time Series for {selected_city}...'):
            X_train, X_val, X_test, Y_train, Y_val, Y_test, inv_scaler = prepare_forecasting_data(df, selected_city)
        
        if X_train is not None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            def inverse_helper(data, scaler):
                shape = data.shape
                data_flat = data.reshape(-1, shape[-1])
                data_rescaled = scaler.inverse_transform(data_flat)
                return data_rescaled.reshape(shape)

            col_lstm, col_trans = st.columns(2)

            # --- LSTM ---
            with col_lstm:
                st.markdown("### Bi-LSTM")
                with st.spinner('Training LSTM...'):
                    lstm_model = train_lstm(X_train, Y_train, X_val, Y_val, input_shape)
                
                Y_pred_scaled = lstm_model.predict(X_test, verbose=0)
                Y_pred_lstm = np.expm1(inverse_helper(Y_pred_scaled, inv_scaler))
                Y_test_actual = np.expm1(inverse_helper(Y_test, inv_scaler))

                mae_lstm = mean_absolute_error(Y_test_actual.flatten(), Y_pred_lstm.flatten())
                rmse_lstm = np.sqrt(mean_squared_error(Y_test_actual.flatten(), Y_pred_lstm.flatten()))
                
                st.metric("MAE", f"{mae_lstm:.2f}", delta_color="inverse")
                st.metric("RMSE", f"{rmse_lstm:.2f}", delta_color="inverse")

            # --- Transformer ---
            with col_trans:
                st.markdown("### Transformer")
                with st.spinner('Training Transformer...'):
                    transformer_model = train_transformer(X_train, Y_train, X_val, Y_val, input_shape)
                
                Y_pred_scaled_t = transformer_model.predict(X_test, verbose=0)
                Y_pred_trans = np.expm1(inverse_helper(Y_pred_scaled_t, inv_scaler))

                mae_trans = mean_absolute_error(Y_test_actual.flatten(), Y_pred_trans.flatten())
                rmse_trans = np.sqrt(mean_squared_error(Y_test_actual.flatten(), Y_pred_trans.flatten()))
                
                st.metric("MAE", f"{mae_trans:.2f}", delta_color="inverse")
                st.metric("RMSE", f"{rmse_trans:.2f}", delta_color="inverse")

            # --- Visualization ---
            st.write("---")
            st.subheader("Visual Comparison")
            
            sample_idx = random.randint(0, len(Y_test_actual) - 1)
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            
            # PM2.5
            ax[0].plot(range(FORECAST_HORIZON), Y_test_actual[sample_idx, :, 0], label='Actual', marker='o', color='black')
            ax[0].plot(range(FORECAST_HORIZON), Y_pred_lstm[sample_idx, :, 0], label='LSTM', linestyle='--', color='red')
            ax[0].plot(range(FORECAST_HORIZON), Y_pred_trans[sample_idx, :, 0], label='Transformer', linestyle='--', color='green')
            ax[0].set_title(f"PM2.5 Forecast (Test Sample #{sample_idx})")
            ax[0].set_ylabel("µg/m³")
            ax[0].set_xlabel("Days Ahead")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)

            # PM10
            ax[1].plot(range(FORECAST_HORIZON), Y_test_actual[sample_idx, :, 1], label='Actual', marker='o', color='black')
            ax[1].plot(range(FORECAST_HORIZON), Y_pred_lstm[sample_idx, :, 1], label='LSTM', linestyle='--', color='red')
            ax[1].plot(range(FORECAST_HORIZON), Y_pred_trans[sample_idx, :, 1], label='Transformer', linestyle='--', color='green')
            ax[1].set_title(f"PM10 Forecast (Test Sample #{sample_idx})")
            ax[1].set_xlabel("Days Ahead")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        else:
            st.error("Insufficient data for this city to train the models.")
    else:
        st.write("👈 Upload your 'city_hour.csv' and click 'Run Models' in the sidebar to start!")