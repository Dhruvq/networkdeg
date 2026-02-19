import pandas as pd
import numpy as np

INPUT_FILE = "network_telemetry.csv"
OUTPUT_FILE = "training_data.csv"
WINDOW_SIZE = '1min'  # Aggregate pings into 1-minute buckets
PREDICTION_HORIZON = 5 # We want to predict 5 minutes into the future

def load_and_filter_data(filepath):
    """
    Loads raw data and selects the SINGLE most active probe.
    (We model one specific connection path, not the global average).
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Count how many pings each probe has
    probe_counts = df['probe_id'].value_counts()
    top_probe = probe_counts.idxmax()
    
    print(f"🔬 Focusing on Probe ID: {top_probe} (Count: {probe_counts[top_probe]})")
    
    # Filter for just this probe
    df_probe = df[df['probe_id'] == top_probe].copy()
    return df_probe.sort_values('timestamp')

def create_features(df):
    """
    Turns raw pings into time-series features (Jitter, Trends, etc.)
    """
    # Set timestamp as index for resampling
    df.set_index('timestamp', inplace=True)
    
    # 1. Resample to 1-minute windows
    # We calculate stats for every minute of data
    df_resampled = df.resample(WINDOW_SIZE).agg({
        'rtt': ['mean', 'std', 'min', 'max', 'count']
    })
    
    # Flatten the multi-level column names (e.g., rtt_mean, rtt_std)
    df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
    
    # 2. Rename for clarity
    df_resampled.rename(columns={
        'rtt_mean': 'avg_latency',
        'rtt_std': 'jitter', # Standard deviation IS Jitter
        'rtt_count': 'packet_count'
    }, inplace=True)
    
    # 3. Handle Missing Data (Forward Fill)
    # If a minute has no pings, assume it's same as last minute (Systems Engineering Best Practice)
    df_resampled = df_resampled.ffill()
    
    # 4. Add "Momentum" Features (Is lag getting worse?)
    df_resampled['latency_change_5m'] = df_resampled['avg_latency'].diff(5) # Change vs 5 mins ago
    
    return df_resampled

def label_data(df):
    """
    Creates the 'Target' variable using Dynamic Z-Score Thresholding.
    """
    # 1. Calculate the Dynamic Threshold (Baseline)
    # What is "normal" for this connection over the last hour?
    df['rolling_mean'] = df['avg_latency'].rolling(window=60, min_periods=1).mean()
    df['rolling_std'] = df['avg_latency'].rolling(window=60, min_periods=1).std()
    
    # Define Degraded as: > 2.5 Standard Deviations above the baseline
    df['threshold'] = df['rolling_mean'] + (2.5 * df['rolling_std'])
    
    # 2. Create the current status label (0 = Good, 1 = Bad)
    df['is_degraded'] = (df['avg_latency'] > df['threshold']).astype(int)
    
    # 3. Create the FUTURE Target (Shift backwards)
    # We want to predict if it WILL be degraded in 5 minutes
    df['target_5m_degraded'] = df['is_degraded'].shift(-PREDICTION_HORIZON)
    
    # Drop the last 5 rows (since they don't have a future target yet)
    df.dropna(subset=['target_5m_degraded'], inplace=True)
    
    return df

if __name__ == "__main__":
    # 1. Load Raw
    print("Loading raw telemetry")
    df_raw = load_and_filter_data(INPUT_FILE)
    
    # 2. Engineer Features
    print("Creating time-series features")
    df_features = create_features(df_raw)
    
    # 3. Label Targets
    print("Labeling Targets")
    df_final = label_data(df_features)
    
    # 4. Save
    df_final.to_csv(OUTPUT_FILE)
    
    print("-" * 30)
    print(f"Saved {len(df_final)} rows to {OUTPUT_FILE}")
    print(f"Class Balance: {df_final['target_5m_degraded'].value_counts(normalize=True)}")
    print("-" * 30)
    print(df_final[['avg_latency', 'jitter', 'threshold', 'is_degraded', 'target_5m_degraded']].tail())