# import pandas as pd
# import xgboost as xgb
# import requests
# import time
# import os

# # CONFIG
# MEASUREMENT_ID = 1001 
# RIPE_API_URL = f"https://atlas.ripe.net/api/v2/measurements/{MEASUREMENT_ID}/results/"
# MODEL_FILE = "tournament_model.json"

# def get_live_prediction():
#     # 1. Fetch last 60 minutes of data (needed for rolling window stats)
#     # We grab a bit more than needed to ensure we have a full hour of history
#     start_time = int(time.time()) - 3600
#     params = {
#         "format": "json",
#         "limit": 2000,   # <--- Reduced from 5000
#         "start": start_time
#     }
    
#     try:
#         response = requests.get(RIPE_API_URL, params=params, timeout=30) 
#         data = response.json()
#     except Exception as e:
#         return {"error": f"RIPE API Error: {str(e)}"}

#     if not data:
#         return {"error": "No data received from RIPE"}

#     # 2. Process Data (Mini-ETL Pipeline)
#     clean_records = []
#     for record in data:
#         if 'result' in record:
#             for attempt in record['result']:
#                 if 'rtt' in attempt:
#                     clean_records.append({
#                         'timestamp': record['timestamp'],
#                         'rtt': attempt['rtt']
#                     })
    
#     if not clean_records:
#         return {"error": "RIPE returned data, but no valid ping results."}

#     df = pd.DataFrame(clean_records)
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
#     df.set_index('timestamp', inplace=True)
    
#     # Resample to 1-min to match training format
#     df_resampled = df.resample('1min').agg({
#         'rtt': ['mean', 'std', 'min', 'max', 'count']
#     })
    
#     # Flatten columns
#     df_resampled.columns = ['avg_latency', 'jitter', 'rtt_min', 'rtt_max', 'packet_count']
    
#     # Handle missing data (Forward Fill)
#     df_resampled = df_resampled.ffill().fillna(0)

#     # 3. Feature Engineering (Must match training EXACTLY)
#     # "momentum" feature
#     df_resampled['latency_change_5m'] = df_resampled['avg_latency'].diff(5)
    
#     # Rolling Stats
#     df_resampled['rolling_mean'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).mean()
#     df_resampled['rolling_std'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).std()
    
#     # We only care about the VERY LAST row (The "Now")
#     current_state = df_resampled.iloc[[-1]].copy()
    
#     # 4. Load Model & Predict
#     if not os.path.exists(MODEL_FILE):
#         return {"error": "Model file not found on server."}

#     model = xgb.XGBClassifier()
#     model.load_model(MODEL_FILE)
    
#     # Select only the columns the model was trained on
#     features = ['avg_latency', 'jitter', 'packet_count', 'rtt_min', 'rtt_max', 
#                 'latency_change_5m', 'rolling_mean', 'rolling_std']
    
#     # Ensure column order matches (XGBoost is picky)
#     try:
#         X_live = current_state[features]
#         # Get Probability of Failure (Class 1)
#         probability = model.predict_proba(X_live)[0][1] 
#     except Exception as e:
#          return {"error": f"Prediction failed: {str(e)}"}
    
#     return {
#         "timestamp": str(current_state.index[0]),
#         "probability": float(probability),
#         "latency": float(current_state['avg_latency'].iloc[0]),
#         "jitter": float(current_state['jitter'].iloc[0])
#     }

import time
import os
import json
import math
import pandas as pd
import xgboost as xgb
import requests
from requests.exceptions import RequestException, Timeout
from typing import Optional

# CONFIG
MEASUREMENT_ID = 1001
RIPE_API_URL = f"https://atlas.ripe.net/api/v2/measurements/{MEASUREMENT_ID}/results/"
MODEL_FILE = "tournament_model.json"

# TIMEOUTS / RETRIES / CACHE
RIPE_API_TIMEOUT = 60           # per-request timeout (seconds)
RIPE_API_RETRIES = 3           # number of attempts
RIPE_BACKOFF_FACTOR = 1.5      # exponential backoff multiplier
RIPE_CACHE_TTL = 300           # seconds to keep in-memory RIPE results (5 minutes)
PERSISTENT_PREDICT_PATH = "/tmp/last_prediction.json"  # fallback persisted prediction

# module-level cache & model holder
_RIPE_CACHE = {"ts": 0, "data": None}
_MODEL = None

def _log(*args, **kwargs):
    print("[inference]", *args, **kwargs, flush=True)

def _persist_prediction(pred: dict):
    try:
        with open(PERSISTENT_PREDICT_PATH, "w") as fh:
            json.dump({"ts": int(time.time()), "prediction": pred}, fh)
            _log("Wrote fallback prediction to", PERSISTENT_PREDICT_PATH)
    except Exception as e:
        _log("Failed to persist prediction:", e)

def _load_persisted_prediction() -> Optional[dict]:
    try:
        if os.path.exists(PERSISTENT_PREDICT_PATH):
            with open(PERSISTENT_PREDICT_PATH, "r") as fh:
                j = json.load(fh)
            return j.get("prediction")
    except Exception as e:
        _log("Failed to read persisted prediction:", e)
    return None

def _load_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("Model file not found on server.")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        _MODEL = model
        _log("Loaded XGBoost model from", MODEL_FILE)
    return _MODEL

def _fetch_ripe_once(start_time):
    params = {"format": "json", "limit": 800, "start": start_time}
    resp = requests.get(RIPE_API_URL, params=params, timeout=RIPE_API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def _fetch_ripe_with_retries(start_time):
    now = int(time.time())
    # return cached if fresh
    if _RIPE_CACHE["data"] is not None and (now - _RIPE_CACHE["ts"] < RIPE_CACHE_TTL):
        _log("Using cached RIPE response")
        return _RIPE_CACHE["data"]
    last_exc = None
    backoff = 1.0
    for attempt in range(1, RIPE_API_RETRIES + 1):
        try:
            _log(f"RIPE fetch attempt {attempt} start={start_time}")
            data = _fetch_ripe_once(start_time)
            if data:
                _RIPE_CACHE["data"] = data
                _RIPE_CACHE["ts"] = int(time.time())
            return data
        except Timeout as e:
            last_exc = e
            _log(f"RIPE request timeout (attempt {attempt}):", e)
        except RequestException as e:
            last_exc = e
            _log(f"RIPE request error (attempt {attempt}):", e)
        except Exception as e:
            last_exc = e
            _log(f"Unexpected RIPE error (attempt {attempt}):", e)
        # backoff before retrying
        time.sleep(backoff)
        backoff *= RIPE_BACKOFF_FACTOR
    # all attempts failed
    raise last_exc

def _normalize_records(raw_data):
    """
    Accepts raw JSON (list/dict) from RIPE and returns a list of records
    with fields: timestamp (epoch seconds) and rtt (float).
    This function is defensive to various shapes returned by API.
    """
    recs = []
    if not raw_data:
        return recs
    # RIPE typically returns a list of measurement result dicts
    if isinstance(raw_data, dict):
        # sometimes the API returns {"results":[...]} — try to find list inside
        for v in raw_data.values():
            if isinstance(v, list):
                raw_data = v
                break
    if not isinstance(raw_data, list):
        return recs

    for item in raw_data:
        # expected: item has 'timestamp' and 'result' array where each attempt has 'rtt'
        ts = item.get("timestamp") or item.get("time")
        # accept timestamp as epoch seconds (int) or string; we'll try to coerce later
        results = item.get("result") or item.get("results") or item.get("probe_results") or []
        if not results and isinstance(item, list):
            # some variants are nested lists
            continue
        # If 'result' is something else, attempt best-effort
        for attempt in results:
            if not isinstance(attempt, dict):
                continue
            # RIPE sometimes has 'rtt' or 'rtt_ms' etc.
            rtt = attempt.get("rtt") or attempt.get("rtt_ms") or attempt.get("value")
            if rtt is None:
                continue
            try:
                rtt_val = float(rtt)
            except Exception:
                continue
            recs.append({"timestamp": ts, "rtt": rtt_val})
    return recs

def _to_dataframe(clean_records):
    # build dataframe defensively
    df = pd.DataFrame(clean_records)
    if df.empty:
        return df
    # try multiple timestamp interpretations
    if 'timestamp' in df.columns:
        # try to parse epoch seconds first
        try:
            df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        except Exception:
            df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # fallback: try ISO parse if epoch parse failed
        if df['timestamp_parsed'].isna().all():
            df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp_parsed'])
        # Drop the original timestamp column to avoid tuple index issue
        df = df.drop(columns=['timestamp'])
        df = df.rename(columns={'timestamp_parsed': 'timestamp'}).set_index('timestamp')
    else:
        return pd.DataFrame()  # no timestamp column - give up
    # ensure rtt numeric
    df['rtt'] = pd.to_numeric(df['rtt'], errors='coerce')
    df = df.dropna(subset=['rtt'])
    return df[['rtt']]

def get_live_prediction():
    """
    Robust get_live_prediction:
    - fetches RIPE data with retries and timeout
    - normalizes results into a dataframe resampled to 1min
    - uses loaded XGBoost model to predict
    - if network/API fails, returns persisted cached prediction if available,
      otherwise returns an error dict.
    """
    # allow forcing dummy via env for quick testing
    if os.environ.get("FORCE_DUMMY") == "1":
        _log("FORCE_DUMMY active")
        dummy = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probability": 0.12,
            "latency": 23.4,
            "jitter": 1.2
        }
        return dummy

    start_time = int(time.time()) - 3600
    try:
        raw = _fetch_ripe_with_retries(start_time)
    except Exception as e:
        _log("All RIPE fetch attempts failed:", e)
        # try to return persisted prediction
        persisted = _load_persisted_prediction()
        if persisted:
            _log("Returning persisted prediction due to RIPE failure")
            return persisted
        return {"error": f"RIPE fetch failed: {str(e)}"}

    if not raw:
        _log("RIPE returned empty payload")
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": "No data received from RIPE"}

    # Normalize and build dataframe
    try:
        clean_records = _normalize_records(raw)
    except Exception as e:
        _log("Normalization error:", e)
        clean_records = []

    if not clean_records:
        _log("No valid rtt records after normalization")
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": "RIPE returned data, but no valid ping results."}

    try:
        df = _to_dataframe(clean_records)
        if df.empty:
            raise ValueError("Dataframe empty after parsing timestamps/rtts")
        # Resample to 1 minute to match training format
        df_resampled = df.resample("1min").agg({'rtt': ['mean', 'std', 'min', 'max', 'count']})
        df_resampled.columns = ['avg_latency', 'jitter', 'rtt_min', 'rtt_max', 'packet_count']
        df_resampled = df_resampled.ffill().fillna(0)
        df_resampled['latency_change_5m'] = df_resampled['avg_latency'].diff(5)
        df_resampled['rolling_mean'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).mean()
        df_resampled['rolling_std'] = df_resampled['avg_latency'].rolling(window=60, min_periods=1).std()
        current_state = df_resampled.iloc[[-1]].copy()
    except Exception as e:
        _log("Data processing error:", e)
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": f"Data processing failed: {str(e)}"}

    # Load model & predict
    try:
        model = _load_model()
    except Exception as e:
        _log("Model load error:", e)
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": str(e)}

    features = ['avg_latency', 'jitter', 'packet_count', 'rtt_min', 'rtt_max',
                'latency_change_5m', 'rolling_mean', 'rolling_std']

    try:
        X_live = current_state[features]
    except Exception as e:
        _log("Feature selection failed:", e)
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": f"Feature selection failed: {str(e)}"}

    try:
        probability = float(model.predict_proba(X_live)[0][1])
    except Exception as e:
        _log("Prediction failed:", e)
        persisted = _load_persisted_prediction()
        if persisted:
            return persisted
        return {"error": f"Prediction failed: {str(e)}"}

    result = {
        "timestamp": str(current_state.index[0]),
        "probability": probability,
        "latency": float(current_state['avg_latency'].iloc[0]),
        "jitter": float(current_state['jitter'].iloc[0])
    }

    # persist last successful prediction for offline fallback
    try:
        _persist_prediction(result)
    except Exception as e:
        _log("Warning: failed to persist prediction:", e)

    return result
