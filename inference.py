import pandas as pd
import xgboost as xgb
import requests
import time
import os

# CONFIG
MEASUREMENT_ID = 1001 
RIPE_API_URL = f"https://atlas.ripe.net/api/v2/measurements/{MEASUREMENT_ID}/results/"
MODEL_FILE = "tournament_model.json"

def get_live_prediction():
    # 1. Fetch last 60 minutes of data (needed for rolling window stats)
    # We grab a bit more than needed to ensure we have a full hour of history
    start_time = int(time.time()) - 3600
    params = {
        "format": "json",
        "limit": 2000,   # <--- Reduced from 5000
        "start": start_time
    }
    
    try:
        response = requests.get(RIPE_API_URL, params=params, timeout=30) 
        data = response.json()
    except Exception as e:
        return {"error": f"RIPE API Error: {str(e)}"}

    if not data:
        return {"error": "No data received from RIPE"}

    # 2. Process Data (Mini-ETL Pipeline)
    clean_records = []
    for record in data:
        if 'result' in record:
            for attempt in record['result']:
                if 'rtt' in attempt:
                    clean_records.append({
                        'timestamp': record['timestamp'],
                        'rtt': attempt['rtt']
                    })
    
    if not clean_records:
        return {"error": "RIPE returned data, but no valid ping results."}

    df = pd.DataFrame(clean_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Resample to 1-min to match training format
    df_resampled = df.resample('1min').agg({
        'rtt': ['mean', 'std', 'min', 'max', 'count']
    })
    
    # Flatten columns
    df_resampled.columns = ['avg_latency', 'jitter', 'rtt_min', 'rtt_max', 'packet_count']
    
    # Handle missing data (Forward Fill)
    df_resampled = df_resampled.ffill().fillna(0)

    # 3. Feature Engineering (Must match training EXACTLY)
    # "momentum" feature
    df_resampled['latency_change_5m'] = df_resampled['avg_latency'].diff(5)
    
    # Rolling Stats
    df_resampled['rolling_mean'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).mean()
    df_resampled['rolling_std'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).std()
    
    # We only care about the VERY LAST row (The "Now")
    current_state = df_resampled.iloc[[-1]].copy()
    
    # 4. Load Model & Predict
    if not os.path.exists(MODEL_FILE):
        return {"error": "Model file not found on server."}

    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # Select only the columns the model was trained on
    features = ['avg_latency', 'jitter', 'packet_count', 'rtt_min', 'rtt_max', 
                'latency_change_5m', 'rolling_mean', 'rolling_std']
    
    # Ensure column order matches (XGBoost is picky)
    try:
        X_live = current_state[features]
        # Get Probability of Failure (Class 1)
        probability = model.predict_proba(X_live)[0][1] 
    except Exception as e:
         return {"error": f"Prediction failed: {str(e)}"}
    
    return {
        "timestamp": str(current_state.index[0]),
        "probability": float(probability),
        "latency": float(current_state['avg_latency'].iloc[0]),
        "jitter": float(current_state['jitter'].iloc[0])
    }